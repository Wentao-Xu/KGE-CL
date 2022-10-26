# Source code and the datasets for KGE-CL

## Dependencies
- Python 3.6+
- PyTorch 1.0+
- NumPy 1.17.2+
- tqdm 4.41.1+

## Reproduce the Results

### 1. Preprocess the Datasets
To preprocess the datasets, run the following commands.

```shell script
cd code
python process_datasets.py
```
Now, the processed datasets are in the data directory

### 2. run KGE-CL
To reproduce the results of KGE-CL on WN18RR, FB15k237 and YAGO3-10,
please run the following commands.

```shell script
#################################### WN18RR ####################################
# RESCAL
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset WN18RR --model RESCAL --rank 512 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 512 --regularizer DURA_RESCAL --reg 1e-1 --max_epochs 200 \
--valid 5 -train -id 0 -save -weight --a_hr 2

# ComplEx
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset WN18RR --model ComplEx --rank 2000 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 200 --regularizer DURA_W --reg 1e-1 --max_epochs 50 \
--valid 5 -train -id 0 -save -weight  --temperature 0.5 --a_tr 2

#################################### FB237 ####################################
# RESCAL
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset FB237 --model RESCAL --rank 512 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 512 --regularizer DURA_RESCAL --reg 5e-2 --max_epochs 200 \
--valid 5 -train -id 0 -save --a_tr 2

# ComplEx
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset FB237 --model ComplEx --rank 2000 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 200 --regularizer DURA_W --reg 5e-2 --max_epochs 200 \
--valid 5 -train -id 0 -save --temperature 0.5 --a_h 2

#################################### YAGO3-10 ####################################
# RESCAL
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset YAGO3-10 --model RESCAL --rank 512 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 512 --regularizer DURA_RESCAL_W --reg 5e-3 --max_epochs 200 \
--valid 5 -train -id 0 -save -weight --a_tr 1

# ComplEx
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset YAGO3-10 --model ComplEx --rank 1000 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 200 --regularizer DURA_W --reg 5e-3 --max_epochs 200 \
--valid 5 -train -id 0 -save --temperature 0.5 --a_t 1
```
