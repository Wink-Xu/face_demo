# -*- coding: utf-8 -*-
"""
created on Sat Apr 13 15:11:34 2019

@author: Wink
"""

import os
import os.path as path
from os.path import join,exists,basename
import json
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import cv2
from multiprocessing import Manager, Pool
import time
import sys
import argparse
import pdb


def tform_maker(std_points, feat_points, points_num):
    EPS = 1e-4
    sum_x = 0
    sum_y = 0
    sum_u = 0
    sum_v = 0
    sum_xx_yy = 0
    sum_ux_vy = 0
    sum_vx__uy = 0
    tform = [0, 0, 0, 0, 0, 0]

    for c in range(points_num):
        x_off = c * 2;
        y_off = x_off + 1;
        sum_x += std_points[c * 2];
        sum_y += std_points[c * 2 + 1];
        sum_u += feat_points[x_off];
        sum_v += feat_points[y_off];
        sum_xx_yy += std_points[c * 2] * std_points[c * 2]+std_points[c * 2 + 1] * std_points[c * 2 + 1]
        sum_ux_vy += std_points[c * 2] * feat_points[x_off]+std_points[c * 2 + 1] * feat_points[y_off]
        sum_vx__uy += feat_points[y_off] * std_points[c * 2] -feat_points[x_off] * std_points[c * 2 + 1]

    if sum_xx_yy <= EPS:
        return 0,tform

    q = sum_u - sum_x * sum_ux_vy / sum_xx_yy + sum_y * sum_vx__uy / sum_xx_yy
    p = sum_v - sum_y * sum_ux_vy / sum_xx_yy - sum_x * sum_vx__uy / sum_xx_yy
    r = points_num - (sum_x * sum_x + sum_y * sum_y) / sum_xx_yy

    if r < EPS and r > -EPS:
        return 0,tform


    a = (sum_ux_vy - sum_x * q / r - sum_y * p / r) / sum_xx_yy
    b = (sum_vx__uy + sum_y * q / r - sum_x * p / r) / sum_xx_yy
    c = q / r;
    d = p / r;

    tform[0] = a
    tform[4] = a
    tform[1] = -b
    tform[3] = b
    tform[2] = c
    tform[5] = d
    return tform


def align_face(points,img):
    
    std_points = [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ]

    img = np.array(img)
    points=np.array(points).reshape(-1)
    std_points=np.array(std_points).reshape(-1)
    tform = tform_maker(std_points, points, 5)

    tform = np.array(tform).reshape((2, 3))
    dst_img = Image.fromarray(
        np.uint8(cv2.warpAffine(img, tform, (112, 112), flags=(cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR))))
    return dst_img



# Align Face

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Align Face')
    parser.add_argument('-d', '--img_path', type =str, default ='')
    ###  you need get the keypoints firstly
    ### python getKeypoints.py
    args = parser.parse_args()

    print(" ------ Face Align ------")
    imgpath = args.img_path
    keypointspath = os.path.join('../results', os.path.basename(imgpath).replace('.jpg', '.json'))
    with open(keypointspath, 'r') as fp:
        keypoints = json.load(fp)

    kp = keypoints['keypoints'][:10]
    img = Image.open(imgpath)
    dst_img = align_face(kp, img)
    dst_img.save(os.path.join('../results', os.path.basename(imgpath).replace('.jpg', '_align.jpg')))
    print(" ------ Finished ------")



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
