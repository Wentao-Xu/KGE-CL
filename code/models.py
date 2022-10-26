from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
from torch.nn import functional as F, Parameter
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class KBCModel(nn.Module, ABC):
    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :return:
        """
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            b_begin = 0
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size]
                target_idxs = these_queries[:, 2].cpu().tolist()
                scores,_, _ = self.forward(these_queries)
                targets = torch.stack([scores[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                for i, query in enumerate(these_queries):
                    filter_out = filters[(query[0].item(), query[1].item())]
                    filter_out += [queries[b_begin + i, 2].item()]  # Add the tail of this (b_begin + i) query
                    scores[i, torch.LongTensor(filter_out)] = -1e6
                ranks[b_begin:b_begin + batch_size] += torch.sum(
                    (scores >= targets).float(), dim=1
                ).cpu()
                b_begin += batch_size
        return ranks


class RESCAL(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(RESCAL, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(sizes[0], rank, sparse=True),
            nn.Embedding(sizes[1], rank * rank, sparse=True),
        ])

        nn.init.xavier_uniform_(tensor=self.embeddings[0].weight)
        nn.init.xavier_uniform_(tensor=self.embeddings[1].weight)

        self.lhs = self.embeddings[0]
        self.rel = self.embeddings[1]
        self.rhs = self.embeddings[0]

    def forward(self, x, p_tail=None, p_rel=None, p_head=None, p_h_r=None, mod=None):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1]).reshape(-1, self.rank, self.rank)
        rhs = self.rhs(x[:, 2])
        self_hr = (torch.bmm(lhs.unsqueeze(1), rel)).squeeze()
        lhs_proj = lhs.view(-1, self.rank, 1)
        rhs_proj = rhs.view(-1, 1, self.rank)
        lr_proj = torch.bmm(lhs_proj, rhs_proj).view(-1, self.rank * self.rank)

        if mod == 0:
            if p_tail is not None:
                h = self.lhs(p_tail[:, 0])
                r = self.rel(p_tail[:, 1]).reshape(-1, self.rank, self.rank)
                hr = torch.bmm(h.unsqueeze(1), r).squeeze()
                return self_hr, hr
            if p_rel is not None:
                h = self.embeddings[0](p_rel[:, 0])
                t = self.embeddings[0](p_rel[:, 1])
                ht = h * t
                self_ht = rhs * lhs
                return self_ht, ht
            if p_head is not None:
                t = self.embeddings[0](p_head[:, 0])
                r = self.embeddings[1](p_head[:, 1]).reshape(-1, self.rank, self.rank)
                tr = torch.bmm(t.unsqueeze(1), r).squeeze()
                self_tr = torch.bmm(rhs.unsqueeze(1), rel).squeeze()
                return self_tr, tr
        else:
            return self_hr @ self.rhs.weight.t(), lr_proj @ self.rel.weight.t(), [(lhs, rel, rhs)]


class ComplEx(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(ComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

    def forward(self, x, p_tail=None, p_rel=None, p_head=None, p_h_r=None, mod=None, r_weight=None):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        self_hr_R = lhs[0] * rel[0] - lhs[1] * rel[1]
        self_hr_I = lhs[0] * rel[1] + lhs[1] * rel[0]
        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]

        self_ht_R = rhs[0] * lhs[0] + rhs[1] * lhs[1]
        self_ht_I = rhs[0] * lhs[1] - rhs[1] * lhs[0]
        to_score1 = self.embeddings[1].weight
        to_score1 = to_score1[:, :self.rank], to_score1[:, self.rank:]

        if mod == 0:
            if p_tail is not None:
                h = self.embeddings[0](p_tail[:, 0])
                r = self.embeddings[1](p_tail[:, 1])
                h = h[:, :self.rank], h[:, self.rank:]
                r = r[:, :self.rank], r[:, self.rank:]
                hr_R = h[0] * r[0] - h[1] * r[1]
                hr_I = h[0] * r[1] + h[1] * r[0]
                return torch.cat((self_hr_R, self_hr_I), 1), torch.cat((hr_R, hr_I), 1)
            if p_rel is not None:
                h = self.embeddings[0](p_rel[:, 0])
                t = self.embeddings[0](p_rel[:, 1])
                h = h[:, :self.rank], h[:, self.rank:]
                t = t[:, :self.rank], t[:, self.rank:]
                ht_R = h[0] * t[0] + h[1] * t[1]
                ht_I = h[0] * t[1] - h[1] * t[0]
                self_ht_R = rhs[0] *lhs[0] + rhs[1] * lhs[1]
                self_ht_I = rhs[0] * lhs[1] - rhs[1] * lhs[0]
                return torch.cat((self_ht_R, self_ht_I), 1), torch.cat((ht_R, ht_I), 1)
            if p_head is not None:
                t = self.embeddings[0](p_head[:, 0])
                r = self.embeddings[1](p_head[:, 1])
                t = t[:, :self.rank], t[:, self.rank:]
                r = r[:, :self.rank], r[:, self.rank:]
                tr_R = t[0] * r[0] + t[1] * r[1]
                tr_I = t[0] * r[1] - t[1] * r[0]
                self_tr_R = rhs[0] * rel[0] + rhs[1] * rel[1]
                self_tr_I = rhs[0] * rel[1] - rhs[1] * rel[0]
                return torch.cat((self_tr_R, self_tr_I), 1), torch.cat((tr_R, tr_I), 1)
        else:
            return (
                           self_hr_R @ to_score[0].transpose(0, 1) + self_hr_I @ to_score[1].transpose(0, 1)
                   ), (
                           self_ht_R @ to_score1[0].transpose(0, 1) + self_ht_I @ to_score1[1].transpose(0, 1)
                   ), [
                (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))
                   ]


class CL(nn.Module):
    def __init__(self, rank, temperature, hidden_size):
        super().__init__()
        # self.projection = encoder_MLP(rank, hidden_size)
        self.temperature = temperature

    def get_negative_mask(self, batch_size, labels1=None, labels2=None):
        if labels2 is None:
            labels1 = labels1.contiguous().view(-1, 1)
            if labels1.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels1, labels1.T).float().cuda()
        else:
            labels1 = labels1.contiguous().view(-1, 1)
            mask1 = torch.eq(labels1, labels1.T).float().cuda()
            labels2 = labels2.contiguous().view(-1, 1)
            mask2 = torch.eq(labels2, labels2.T).float().cuda()
            mask = mask1*mask2
            mask = mask.float().cuda()
        mask = mask.repeat(2, 2)
        return mask

    def pos_loss(self, self_predictions, pos_predictions, labels1=None, labels2=None):
        pos_predictions = F.normalize(pos_predictions, dim=-1)
        self_predictions = F.normalize(self_predictions, dim=-1)
        mask = self.get_negative_mask(self_predictions.shape[0], labels1, labels2).cuda()
        out = torch.cat([self_predictions, pos_predictions], dim=0)
        similarity_m = F.cosine_similarity(out.unsqueeze(1), out.unsqueeze(0), dim=-1)
        pos = (similarity_m * mask) / self.temperature
        exp_logits = torch.exp(similarity_m / self.temperature)
        pos = pos.sum(1)
        pos = pos
        neg = exp_logits * ((~mask.bool()).float())
        neg = neg.sum(dim=-1)
        pos_loss = (- pos + torch.log(neg)) / mask.sum(-1)
        pos_loss = pos_loss.mean()
        return pos_loss

    def forward(self, x1, x2, labels1=None, labels2=None):
        # x1 = self.projection(x1)
        # x2 = self.projection(x2)
        loss = self.pos_loss(x1, x2, labels1, labels2)
        return loss

