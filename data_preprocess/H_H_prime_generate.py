#!~/anaconda3/bin/python3
# ******************************************************
# Author: Pengshuai Yang
# Last modified: 2020-10-26 08:39
# Email: yps18@mails.tsinghua.edu.cn
# Filename: H_H_prime_generate.py
# Description: 
#   dataset preprocessing
# ******************************************************

import os
import cv2
import random
import numpy as np
import threadpool
import spams
from csco_vahadane import vahadane, read_image


PATCH_PATH = './train/patches/'
H_PATH = './train/H/'
E_PATH = './train/E/'
H_prime_PATH = './train/H_prime/'
E_prime_PATH = './train/E_prime/'

IMAGE_SIZE = 224

'''
def get_HorE(W, H):
    
    img = np.dot(W.reshape(3,1), H.reshape(1,-1))
    img = np.clip(255 * np.exp(-1*img),0,255)
    img = img.T.reshape(IMAGE_SIZE,IMAGE_SIZE, 3).astype(np.uint8)
    return img 
'''
def get_HorE(concentration):
    return np.clip(255*np.exp(-1*concentration),0,255).reshape(IMAGE_SIZE, IMAGE_SIZE).astype(np.uint8)

'''
def save_img(W_matrix, H_matrix, name):
    H = get_HorE(W_matrix[:,0], H_matrix[0,:])
    E = get_HorE(W_matrix[:,1], H_matrix[1,:])
    
    cv2.imwrite(H_PATH+name+'_H.png', cv2.cvtColor(H, cv2.COLOR_RGB2BGR))
    cv2.imwrite(E_PATH+name+'_E.png', cv2.cvtColor(E, cv2.COLOR_RGB2BGR))

def save_img_prime(W_matrix, H_matrix, name):
    H = get_HorE(W_matrix[:,0], H_matrix[0,:])
    E = get_HorE(W_matrix[:,1], H_matrix[1,:])
    
    cv2.imwrite(H_prime_PATH+name+'_H_prime.png', cv2.cvtColor(H, cv2.COLOR_RGB2BGR))
    cv2.imwrite(E_prime_PATH+name+'_E_prime.png', cv2.cvtColor(E, cv2.COLOR_RGB2BGR))

'''

def save_img(concen, concen_prime, name):
    H = get_HorE(concen[0,:])
    cv2.imwrite(H_PATH+name+'_H.png', H)
    E = get_HorE(concen[1,:])
    cv2.imwrite(E_PATH+name+'_E.png', E)
    H_prime = get_HorE(concen_prime[0,:])
    cv2.imwrite(H_prime_PATH+name+'_H_prime.png', H_prime)
    E_prime = get_HorE(concen_prime[1,:])
    cv2.imwrite(E_prime_PATH+name+'_E_prime.png', E_prime)


def get_img_list(path, k=None):
    img_list = os.listdir(PATCH_PATH)
    random.seed(10)
    random.shuffle(img_list)
    if k!=None:
        return img_list[:k]
    else:
        return img_list 

def main(img):
    name = img.split('.')[0]
    print(name)
    img = read_image(PATCH_PATH+img)
    #W_m, H_m = vhd.stain_separate(img)
    stain, concen = vhd.stain_separate(img)
    #save_img(stain, concen, name)
    perturb_stain = stain + np.random.randn(3,2)*0.05
    perturb_stain, perturb_concen = vhd.stain_separate(img, perturb_stain)
    save_img(concen, perturb_concen, name)
    #save_img_prime(perturb_stain, perturb_concen, name)


if __name__ == '__main__':
    k = None
    vhd = vahadane(LAMBDA1=0.01, LAMBDA2=0.01, fast_mode=0, getH_mode=1,
                   ITER=50)
    vhd.show_config()
    img_list = get_img_list(PATCH_PATH, k)
    
    pool = threadpool.ThreadPool(8)
    requests = threadpool.makeRequests(main, img_list)
    [pool.putRequest(req) for req in requests]
    pool.wait()
        

