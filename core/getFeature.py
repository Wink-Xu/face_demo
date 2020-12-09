import os
from os.path import join
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np
import pickle
from torch.nn import DataParallel
import math
import random
import cv2
import argparse
import torchvision.utils as vutils
from torch.autograd import Variable
import pdb
import matplotlib.pylab as plt

#import model.single_DR_GAN_model as single_model
#from model.weights import init_weights
import torchvision.transforms as transforms
#from generate_one_image import *
import json

import argparse


#输入只和网络相关,Tensor
def getfeature(input,model):
    tmp=torch.zeros(input.size()[0]).long().cuda()
    y,features=model(input,tmp)
    return features.cpu().data.numpy()


def loadmodel(modelpath, use_gpu):
    if use_gpu:       
        model=torch.load(modelpath)
        # if len(args.gpu_ids)>1:
        #     model=DataParallel(model)
    #  model = model.module
        model.cuda()
        model.eval()
    else:
        model = torch.load(modelpath, map_location=torch.device('cpu'))
        device = torch.device('cpu')
        model.to(device)
        model.eval()
    return model



def get_faceFeature(img):
    use_gpu = 0
    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        device = torch.device('cpu')
    modelpath='../model/arcface_r100_ms1m_atthead2_7_trans.pkl'

    model=loadmodel(modelpath, use_gpu)

    imgarr=np.zeros((1,3,112,112))
    img = img.resize((112, 112))
        #img=moveBlur(img)
    img=np.array(img,dtype=np.float32).swapaxes(1, 2).swapaxes(0, 1).reshape(1,3,112,112)
    img=np.flip(img,axis=0).copy()
    img=np.flip(img,axis=2).copy()
    imgarr = (img-127.5)/128
    if use_gpu:
        input = torch.Tensor(imgarr).float().cuda()
    else:
        input=torch.Tensor(imgarr).float().to(device)

    features_out = getfeature(input, model)
    #import pdb; pdb.set_trace()
    return features_out

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Extract Feature')
    parser.add_argument('-d', '--img_path', type=str, default ='' )
    parser.add_argument('--gpu_ids', type=str, default = '0')
    args = parser.parse_args()

    print(" ------ Get Face Feature ------")
    img_path = args.img_path

    img=Image.open(img_path)
    feature = get_faceFeature(img)

   # print(feature)
    with open(os.path.join('../results', os.path.basename(img_path).replace('.jpg', '.vec')), 'w') as fw:
        for i in range(len(feature[0])):
            fw.write(str(feature[0][i]))
            fw.write(' ')
   
    print(" ------ Finished ------")




