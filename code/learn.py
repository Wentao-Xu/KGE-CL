import os
import json
import argparse
import numpy as np
import pickle

import torch
from torch import optim
from torch import nn
from datasets import Dataset
from models import *
from regularizers import *
from optimizers import KBCOptimizer

# from random import choice
datasets = ['WN18RR', 'FB237', 'YAGO3-10']

parser = argparse.ArgumentParser(
    description="Tensor Factorization for Knowledge Graph Completion"
)

parser.add_argument(
    '--dataset', choices=datasets,
    help="Dataset in {}".format(datasets)
)

parser.add_argument(
    '--model', type=str, default='CP'
)

parser.add_argument(
    '--regularizer', type=str, default='NA',
)

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument(
    '--optimizer', choices=optimizers, default='Adagrad',
    help="Optimizer in {}".format(optimizers)
)

parser.add_argument(
    '--max_epochs', default=50, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid', default=3, type=float,
    help="Number of epochs before valid."
)
parser.add_argument(
    '--rank', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--reg', default=0, type=float,
    help="Regularization weight"
)
parser.add_argument(
    '--init', default=1e-3, type=float,
    help="Initial scale"
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--decay1', default=0.9, type=float,
    help="decay rate for the first moment estimate in Adam"
)
parser.add_argument(
    '--decay2', default=0.999, type=float,
    help="decay rate for second moment estimate in Adam"
)
parser.add_argument('--name', type=str, default='WN18RR')
parser.add_argument('-train', '--do_train', action='store_true')
parser.add_argument('-test', '--do_test', action='store_true')
parser.add_argument('-save', '--do_save', action='store_true')
parser.add_argument('-weight', '--do_ce_weight', action='store_true')
parser.add_argument('-path', '--save_path', type=str, default='../logs/')
parser.add_argument('-id', '--model_id', type=str, default='0')
parser.add_argument('-ckpt', '--checkpoint', type=str, default='')
parser.add_argument(
    '--negative_sample_size', default=200, type=int,
    help="negative sample size"
)
parser.add_argument('--temperature', default=0.9, type=float, help="temperature")
parser.add_argument('--out_size', default=4000, type=int, help="out size")
parser.add_argument('--a_h', default=0, type=float, help="a_h")
parser.add_argument('--a_t', default=0, type=float, help="a_t")
parser.add_argument('--a_hr', default=0, type=float, help="a_hr")
parser.add_argument('--a_tr', default=0, type=float, help="a_tr")
args = parser.parse_args()

if args.do_save:
    assert args.save_path
    save_suffix = args.model + '_' + args.regularizer + '_' + args.dataset + '_' + args.model_id

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    save_path = os.path.join(args.save_path, save_suffix)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

# data_path = "../data"
data_path = os.path.abspath(os.curdir) + "../data"
dataset = Dataset(data_path, args.dataset)
examples = torch.from_numpy(dataset.get_train().astype('int64'))
dic_tr, dic_hr, dic_h, dic_t = dataset.get_pos(examples)
if args.do_ce_weight:
    ce_weight = torch.Tensor(dataset.get_weight()).cuda()
else:
    ce_weight = None

print(dataset.get_shape())

model = None
regularizer = None
exec('model = ' + args.model + '(dataset.get_shape(), args.rank, args.init)')
exec('regularizer = ' + args.regularizer + '(args.reg)')
regularizer = [regularizer, N3(args.reg)]

device = torch.device('cuda')
model.to(device)
for reg in regularizer:
    reg.to(device)
optim_method = {
    'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
    'Adam': lambda: optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.decay1, args.decay2)),
    'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate)
}[args.optimizer]()

optimizer = KBCOptimizer(
    datasets, args.dataset, args.model, model, regularizer, optim_method, args.batch_size, args.temperature,
    args.rank, args.out_size, args.a_h, args.a_t, args.a_hr, args.a_tr
)


def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
    :param hits:
    :return:
    """
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MRR': m, 'hits@[1,3,10]': h}, m


cur_loss = 0
base_mrr = 0
test_res = None
if args.checkpoint is not '':
    model.load_state_dict(torch.load(os.path.join(args.checkpoint, 'checkpoint'), map_location='cuda:0'))


def save_args():
    torch.save(model.state_dict(), os.path.join(save_path, 'checkpoint'))
    embeddings = model.embeddings
    len_emb = len(embeddings)
    if len_emb == 2:
        np.save(os.path.join(save_path, 'entity_embedding.npy'), embeddings[0].weight.detach().cpu().numpy())
        np.save(os.path.join(save_path, 'relation_embedding.npy'), embeddings[1].weight.detach().cpu().numpy())
    elif len_emb == 3:
        np.save(os.path.join(save_path, 'head_entity_embedding.npy'), embeddings[0].weight.detach().cpu().numpy())
        np.save(os.path.join(save_path, 'relation_embedding.npy'), embeddings[1].weight.detach().cpu().numpy())
        np.save(os.path.join(save_path, 'tail_entity_embedding.npy'), embeddings[2].weight.detach().cpu().numpy())
    else:
        print('SAVE ERROR!')
    return 1


if args.do_train:
    with open(os.path.join(save_path, 'train.log'), 'w') as log_file:
        for e in range(args.max_epochs):
            print("Epoch: {}".format(e + 1))

            cur_loss = optimizer.epoch(
                examples, e=e, weight=ce_weight, dic_tr=dic_tr, dic_t=dic_t, dic_hr=dic_hr, dic_h=dic_h)

            if (e + 1) % args.valid == 0:
                (valid, _), (test, _), (train, _) = [
                    avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
                    for split in ['valid', 'test', 'train']
                ]

                print("\t TRAIN: ", train)
                print("\t VALID: ", valid)
                print("\t TEST: ", test)

                log_file.write("Epoch: {}\n".format(e + 1))
                log_file.write("\t TRAIN: {}\n".format(train))
                log_file.write("\t VALID: {}\n".format(valid))
                log_file.write("\t TEST: {}\n".format(test))
                log_file.flush()
            # print("\t TEST : ", test_res)
