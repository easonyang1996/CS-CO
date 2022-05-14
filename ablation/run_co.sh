CUDA_VISIBLE_DEVICES=5,6,7 python3 -u ./4_csco_train.py configs/cs-co_conf_resnet18_10.0.ini 3
CUDA_VISIBLE_DEVICES=5,6,7 python3 -u ./4_csco_train.py configs/cs-co_conf_resnet18_5.0.ini 3
CUDA_VISIBLE_DEVICES=5,6,7 python3 -u ./4_csco_train.py configs/cs-co_conf_resnet18_1.0.ini 3
CUDA_VISIBLE_DEVICES=5,6,7 python3 -u ./4_csco_train.py configs/cs-co_conf_resnet18_0.5.ini 3
CUDA_VISIBLE_DEVICES=5,6,7 python3 -u ./4_csco_train.py configs/cs-co_conf_resnet18_0.1.ini 3


CUDA_VISIBLE_DEVICES=5,6,7 python3 -u ./4_csco_train.py configs/cs-co_conf_resnet18_unfix.ini 3
CUDA_VISIBLE_DEVICES=5,6,7 python3 -u ./4_csco_train.py configs/cs-co_conf_resnet18_no_svp.ini 3

CUDA_VISIBLE_DEVICES=1,3,4,5,7,9 python3 -u ./4_csco_train.py configs/cs-co_conf_resnet50.ini 6

