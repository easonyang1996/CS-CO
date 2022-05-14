python3 -u linear_train.py -m cs-co -b resnet50 -d 100
python3 -u linear_train.py -m cs-co -b resnet50 -d 1000
python3 -u linear_train.py -m cs-co -b resnet50 -d 10000

python3 -u linear_train.py -m cs-co -w /home/yangpengshuai/MedIA_SI/nct_gamma_ablation_v2/checkpoint/gamma2_10.0_0.001/csco_cs-co_Adam-step_None_32_0.001_1e-06_1.0_10.0_24_0.41650.pth -b resnet18 -d 100
python3 -u linear_train.py -m cs-co -w /home/yangpengshuai/MedIA_SI/nct_gamma_ablation_v2/checkpoint/gamma2_10.0_0.001/csco_cs-co_Adam-step_None_32_0.001_1e-06_1.0_10.0_24_0.41650.pth -b resnet18 -d 1000
python3 -u linear_train.py -m cs-co -w /home/yangpengshuai/MedIA_SI/nct_gamma_ablation_v2/checkpoint/gamma2_10.0_0.001/csco_cs-co_Adam-step_None_32_0.001_1e-06_1.0_10.0_24_0.41650.pth -b resnet18 -d 10000


#python3 -u linear_train.py -m cs-co -w /home/yangpengshuai/MedIA_SI/nct_gamma_ablation_v2/checkpoint/gamma2_5.0_0.001/csco_cs-co_Adam-step_None_32_0.001_1e-06_1.0_5.0_30_0.21305.pth -b resnet18 -d 100
#python3 -u linear_train.py -m cs-co -w /home/yangpengshuai/MedIA_SI/nct_gamma_ablation_v2/checkpoint/gamma2_5.0_0.001/csco_cs-co_Adam-step_None_32_0.001_1e-06_1.0_5.0_30_0.21305.pth -b resnet18 -d 1000
#python3 -u linear_train.py -m cs-co -w /home/yangpengshuai/MedIA_SI/nct_gamma_ablation_v2/checkpoint/gamma2_5.0_0.001/csco_cs-co_Adam-step_None_32_0.001_1e-06_1.0_5.0_30_0.21305.pth -b resnet18 -d 10000


#python3 -u linear_train.py -m cs-co -w /home/yangpengshuai/MedIA_SI/nct_gamma_ablation_v2/checkpoint/gamma2_1.0_0.001/csco_cs-co_Adam-no_None_32_0.001_1e-06_1.0_1.0_13_0.06624.pth -b resnet18 -d 100
#python3 -u linear_train.py -m cs-co -w /home/yangpengshuai/MedIA_SI/nct_gamma_ablation_v2/checkpoint/gamma2_1.0_0.001/csco_cs-co_Adam-no_None_32_0.001_1e-06_1.0_1.0_13_0.06624.pth -b resnet18 -d 1000
#python3 -u linear_train.py -m cs-co -w /home/yangpengshuai/MedIA_SI/nct_gamma_ablation_v2/checkpoint/gamma2_1.0_0.001/csco_cs-co_Adam-no_None_32_0.001_1e-06_1.0_1.0_13_0.06624.pth -b resnet18 -d 10000


#python3 -u linear_train.py -m cs-co -w /home/yangpengshuai/MedIA_SI/nct_gamma_ablation_v2/checkpoint/gamma2_0.5_0.001/csco_cs-co_Adam-no_None_32_0.001_1e-06_1.0_0.5_21_0.03917.pth -b resnet18 -d 100
#python3 -u linear_train.py -m cs-co -w /home/yangpengshuai/MedIA_SI/nct_gamma_ablation_v2/checkpoint/gamma2_0.5_0.001/csco_cs-co_Adam-no_None_32_0.001_1e-06_1.0_0.5_21_0.03917.pth -b resnet18 -d 1000
#python3 -u linear_train.py -m cs-co -w /home/yangpengshuai/MedIA_SI/nct_gamma_ablation_v2/checkpoint/gamma2_0.5_0.001/csco_cs-co_Adam-no_None_32_0.001_1e-06_1.0_0.5_21_0.03917.pth -b resnet18 -d 10000


#python3 -u linear_train.py -m cs-co -w /home/yangpengshuai/MedIA_SI/nct_gamma_ablation_v2/checkpoint/gamma2_0.1_0.001/csco_cs-co_Adam-no_None_32_0.001_1e-06_1.0_0.1_19_0.02028.pth -b resnet18 -d 100
#python3 -u linear_train.py -m cs-co -w /home/yangpengshuai/MedIA_SI/nct_gamma_ablation_v2/checkpoint/gamma2_0.1_0.001/csco_cs-co_Adam-no_None_32_0.001_1e-06_1.0_0.1_19_0.02028.pth -b resnet18 -d 1000
#python3 -u linear_train.py -m cs-co -w /home/yangpengshuai/MedIA_SI/nct_gamma_ablation_v2/checkpoint/gamma2_0.1_0.001/csco_cs-co_Adam-no_None_32_0.001_1e-06_1.0_0.1_19_0.02028.pth -b resnet18 -d 10000



########### unfix
#python3 -u linear_train.py -m cs-co -w /home/yangpengshuai/MedIA_SI/nct_gamma_ablation_v2/checkpoint/unfix/csco_cs-co_Adam-no_None_32_0.001_1e-06_1.0_10.0_32_0.41703.pth -b resnet18 -d 100
#python3 -u linear_train.py -m cs-co -w /home/yangpengshuai/MedIA_SI/nct_gamma_ablation_v2/checkpoint/unfix/csco_cs-co_Adam-no_None_32_0.001_1e-06_1.0_10.0_32_0.41703.pth -b resnet18 -d 1000
#python3 -u linear_train.py -m cs-co -w /home/yangpengshuai/MedIA_SI/nct_gamma_ablation_v2/checkpoint/unfix/csco_cs-co_Adam-no_None_32_0.001_1e-06_1.0_10.0_32_0.41703.pth -b resnet18 -d 10000

########### nosvp
#python3 -u linear_train.py -m cs-co -w /home/yangpengshuai/MedIA_SI/nct_gamma_ablation_v2/checkpoint/nosvp/csco_cs-co_Adam-no_None_32_0.001_1e-06_1.0_10.0_9_0.39993.pth -b resnet18 -d 100
#python3 -u linear_train.py -m cs-co -w /home/yangpengshuai/MedIA_SI/nct_gamma_ablation_v2/checkpoint/nosvp/csco_cs-co_Adam-no_None_32_0.001_1e-06_1.0_10.0_9_0.39993.pth -b resnet18 -d 1000
#python3 -u linear_train.py -m cs-co -w /home/yangpengshuai/MedIA_SI/nct_gamma_ablation_v2/checkpoint/nosvp/csco_cs-co_Adam-no_None_32_0.001_1e-06_1.0_10.0_9_0.39993.pth -b resnet18 -d 10000
