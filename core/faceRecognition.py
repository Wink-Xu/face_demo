import os
import numpy as np
import argparse
from PIL import Image
import sys

from getKeypoint import get_keypoint
from alignFace import align_face
from getFeature import get_faceFeature

def cos_distance(f1, f2):

    f1_norm = np.linalg.norm(f1)
    f2_norm = np.linalg.norm(f2)

    score = np.dot(f1, f2.T)/ (f1_norm * f2_norm)

    return score

parser = argparse.ArgumentParser(description='Align Face')
parser.add_argument('-d1', '--img_first', type =str, default ='')
parser.add_argument('-d2', '--img_second', type =str, default ='')

if __name__ == '__main__':

    args = parser.parse_args()
    img_path1 = args.img_first
    img_path2 = args.img_second
### img1  
    img1 = Image.open(img_path1)
    kp_list, _ = get_keypoint(img1)
    dst_img1 = align_face(kp_list, img1)
    feature1 = get_faceFeature(dst_img1)
### img2
    img2 = Image.open(img_path2)
    kp_list, _ = get_keypoint(img2)
    dst_img2 = align_face(kp_list, img2)
    feature2 = get_faceFeature(dst_img2)

    score = cos_distance(feature1, feature2)[0][0]
    
    print("Two image's similarity is {}".format((score+ 1)/2 * 100))
