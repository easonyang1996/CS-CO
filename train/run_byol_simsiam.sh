CUDA_VISIBLE_DEVICES=2,3,4 python3 -u ./1_byol_train_dist.py ./configs/NCT_CRC/byol_conf_resnet18.ini 3
CUDA_VISIBLE_DEVICES=2,3,4 python3 -u ./1_byol_train_dist.py ./configs/NCT_CRC/simsiam_conf_resnet18.ini 3
#CUDA_VISIBLE_DEVICES=1,3,4,5,7,9 python3 -u ./1_byol_train_dist.py ./configs/NCT_CRC/byol_conf_resnet50.ini 6 
#CUDA_VISIBLE_DEVICES=1,3,4,5,7,9 python3 -u ./1_byol_train_dist.py ./configs/NCT_CRC/simsiam_conf_resnet50.ini 6


#CUDA_VISIBLE_DEVICES=1,3,4,5,7,9 python3 -u ./1_byol_train_dist.py ./configs/TCGA_LIHC/simsiam_conf_resnet50.ini 6
#CUDA_VISIBLE_DEVICES=1,3,4,5,7,9 python3 -u ./1_byol_train_dist.py ./configs/xiangya/simsiam_conf_resnet50.ini 6
