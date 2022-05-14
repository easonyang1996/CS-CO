# CS-CO
CS-CO is a hybrid self-supervised visual representation learning method tailored for H&E-stained histopathological images. This work has been presented in MICCAI2021 conference as an oral talk.

MICCAI2021 Paper: [Self-supervised visual representation learning for histopathological images](https://link.springer.com/chapter/10.1007/978-3-030-87196-3_5)

![framework](https://github.com/easonyang1996/CS-CO/blob/main/figs/framework.png)

# Instructions
We provide detailed step-by-step instructions for reproducing experiments of the proposed method on NCT-CRC-HE-100K. You can also run the proposed method on your own dataset in a similar way.

**Step 1** Prepare the dataset.
Please download the dataset from [NCT-CRC-HE-100K](https://zenodo.org/record/1214456#.Yn9lVy8RrfY). NCT-CRC-HE-100K.zip is used as training set and CRC-VAL-HE-7K.zip is used as test set. As mentioned in our paper, we drop images belonging to "BACK" class.


**Step 2** Train the model.


```
python3 
```

**Step 3** Test the model.


```
python3 
```


# Citation

**Yang, P.**, Hong, Z., Yin, X., Zhu, C., Jiang, R. (2021). Self-supervised Visual Representation Learning for Histopathological Images. In: , et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2021. MICCAI 2021. Lecture Notes in Computer Science(), vol 12902. Springer, Cham. https://doi.org/10.1007/978-3-030-87196-3_5


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
