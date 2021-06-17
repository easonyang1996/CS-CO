#!/usr/local/bin/python3
# ******************************************************
# Author: Pengshuai Yang
# Last modified: 2021-01-18 14:00
# Email: 244090225@qq.com
# Filename: csco_utils.py
# Description: 
# 
# ******************************************************
import cv2
import numpy as np


def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # opencv default color space is BGR, change it to RGB
    p = np.percentile(img, 90)
    img = np.clip(img * 255.0 / p, 0, 255).astype(np.uint8)
    return img
