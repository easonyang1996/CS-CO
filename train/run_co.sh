CUDA_VISIBLE_DEVICES=0,1,2 python3 -u ./4_csco_train_dist.py configs/NCT_CRC/cs-co_conf_resnet18.ini 3
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python3 -u ./4_csco_train_dist.py configs/NCT_CRC/cs-co_conf_resnet50.ini 8

#CUDA_VISIBLE_DEVICES=5,6,7 python3 -u ./4_csco_train_dist.py configs/TCGA_LIHC/cs-co_conf.ini 3
#CUDA_VISIBLE_DEVICES=2,3,4 python3 -u ./4_csco_train_dist.py configs/xiangya/cs-co_conf.ini 3
