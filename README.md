# CS-CO
CS-CO is a hybrid self-supervised visual representation learning method tailored for H&E-stained histopathological images. This work has been presented in MICCAI2021 conference as an oral talk. An extended version of the MICCAI2021 paper has been accepted by Medical Image Analysis.

MICCAI2021 Paper: [Self-supervised visual representation learning for histopathological images](https://link.springer.com/chapter/10.1007/978-3-030-87196-3_5)

MedIA Paper: [CS-CO: A Hybrid Self-Supervised Visual Representation Learning Method for H&E-stained Histopathological Images](https://www.sciencedirect.com/science/article/pii/S1361841522001864)

![framework](https://github.com/easonyang1996/CS-CO/blob/main/figs/framework.png)

# Instructions
We provide detailed step-by-step instructions for reproducing experiments of the proposed method on NCT-CRC-HE-100K. You can also run the proposed method on your own dataset in a similar way.

**1. Prepare the dataset**

Please download the dataset from [NCT-CRC-HE-100K](https://zenodo.org/record/1214456#.Yn9lVy8RrfY). In our paper, `NCT-CRC-HE-100K.zip` is used as the training set and `CRC-VAL-HE-7K.zip` is used as the test set. For each set, we exclude images belonging to "BACK" class and move the rest images to one folder. 

```
____NCT-CRC
    |____train
         |____patches
              |____aaa.png
              |____bbb.png
              |____ccc.png
    |____test
         |____patches
              |____ddd.png
              |____eee.png
              |____fff.png
```

Then, please go to `./data_preprocess/` and do stain separation on training and test set separately by running:

```
python3 H_H_prime_generate.py
```

**2. Training**

The training of the proposed CS-CO contains two stages. Please go to `./train/`.

At the first stage, cross-stain prediction can be done by running:

```
CUDA_VISIBLE_DEVICES=0 python3 -u 4_csco_train.py configs/NCT_CRC/cs_conf_resnet18.ini 
```

At the second stage, contrastive learning can be done by running:

```
CUDA_VISIBLE_DEVICES=0,1,2 python3 -u 4_csco_train_dist.py configs/NCT_CRC/cs-co_conf_resnet18.ini 3
```

You can also reproduce competing methods by running `run_byol_simsiam.sh` or `run_chen_xie.sh`. 

The configurations of all methods are saved in `./train/config/`, and you can modify them as you wish.


**3. Evaluation**

For evaluation, we run 10 times Monte Carlo cross-validation under the linear evaluation protocol. Please go to `./linear_eval/` and run:

```
./eval.sh
```

The model weights of Transpath and SimCLR-ciga can be found from [Xiyue-Wang/TransPath](https://github.com/Xiyue-Wang/TransPath) and [ozanciga/self-supervised-histopathology](https://github.com/ozanciga/self-supervised-histopathology) respectively. Please download and put them in `./linear_eval/` if you want to test the performance of Transpath and SimCLR-ciga.

**4. Ablation**

The code of our ablation study can be found in `./ablation/`. You can reproduce ablated models by running:

```
./run_co.sh
```

and evaluate them by running:
```
./eval.sh
```


# Citation

Please use below to cite our paper if you find our work useful in your research.


Yang, P., Hong, Z., Yin, X., Zhu, C. and Jiang, R., 2021, September. Self-supervised visual representation learning for histopathological images. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 47-57). Springer, Cham.

Yang, P., Yin, X., Lu, H., Hu, Z., Zhang, X., Jiang, R. and Lv, H., 2022. CS-CO: A Hybrid Self-Supervised Visual Representation Learning Method for H&E-stained Histopathological Images. Medical Image Analysis, 81, p.102539.



```
@InProceedings{10.1007/978-3-030-87196-3_5,
author="Yang, Pengshuai
and Hong, Zhiwei
and Yin, Xiaoxu
and Zhu, Chengzhan
and Jiang, Rui",
title="Self-supervised Visual Representation Learning for Histopathological Images",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2021",
year="2021",
publisher="Springer International Publishing",
address="Cham",
pages="47--57",
isbn="978-3-030-87196-3"
}
```

```
@article{YANG2022102539,
title = {CS-CO: A Hybrid Self-Supervised Visual Representation Learning Method for H&E-stained Histopathological Images},
journal = {Medical Image Analysis},
volume = {81},
pages = {102539},
year = {2022},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2022.102539},
url = {https://www.sciencedirect.com/science/article/pii/S1361841522001864},
author = {Pengshuai Yang and Xiaoxu Yin and Haiming Lu and Zhongliang Hu and Xuegong Zhang and Rui Jiang and Hairong Lv},
}
```

