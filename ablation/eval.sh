CUDA_VISIBLE_DEVICES=1 python3 linear_train.py -m unfix -d 100
CUDA_VISIBLE_DEVICES=1 python3 linear_train.py -m no_svp -d 100
CUDA_VISIBLE_DEVICES=1 python3 linear_train.py -m cs -d 100
CUDA_VISIBLE_DEVICES=1 python3 linear_train.py -m co -d 100
CUDA_VISIBLE_DEVICES=1 python3 linear_train.py -m unfix -d 1000
CUDA_VISIBLE_DEVICES=1 python3 linear_train.py -m no_svp -d 1000
CUDA_VISIBLE_DEVICES=1 python3 linear_train.py -m cs -d 1000
CUDA_VISIBLE_DEVICES=1 python3 linear_train.py -m co -d 1000
CUDA_VISIBLE_DEVICES=1 python3 linear_train.py -m unfix -d 10000
CUDA_VISIBLE_DEVICES=1 python3 linear_train.py -m no_svp -d 10000
CUDA_VISIBLE_DEVICES=1 python3 linear_train.py -m cs -d 10000
CUDA_VISIBLE_DEVICES=1 python3 linear_train.py -m co -d 10000

