# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:19:11 2019

@author: Wink
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os.path as path
from PIL import Image, ImageDraw, ImageFilter
import shutil
from tqdm import tqdm
import sys
import json
import math
import matplotlib.pylab as plt
import os
from KPNet_FullConv_RGB import KPNet_FullConv_RGB_8x8

import argparse



def output2KP(output):
    batch_size=output.size()[0]
    # output = output.view((batch_size, 15, 3, 3))
    output = output.cpu().data.numpy()
    result=[]
    for index in range(batch_size):
        max_p = [0, 0, 0, 0, 0]
        max_p_xy = [[], [], [], [], []]
        max_p_xy_tmp = [[], [], [], [], []]
        # max_tmp=[0.63, 0.61, 0.67, 0.29, 0.18]
        for p in range(8):
            for q in range(8):
                for r in range(5):
                    v = output[index, r * 3, p, q]
                    # if v>max_tmp[r]:
                    #     print('???')
                    #     continue
                    if v > max_p[r]:
                        max_p[r] = v
                        max_p_xy[r] = [p * 16 + output[index, r * 3 + 1, p, q] * 16,
                                       q * 16 + output[index, r * 3 + 2, p, q] * 16]
                        max_p_xy_tmp[r] = [output[index, r * 3 + 1, p, q], output[index, r * 3 + 2, p, q]]
        score = round(sum(max_p), 4)
        # print(max_p)
        kp=[]
        for i in range(5):
            # print(max_p_xy[i][0],max_p_xy[i][1])
            x = int(max_p_xy[i][0])
            y = int(max_p_xy[i][1])
            kp.append((x,y))
        kp.append(score)
        result.append(kp)
    return result


def get_keypoint(img, imgorg_size = [96, 96, 32, 32], factor = 100):
    """
    func:       提取图片中人脸的关键点
    factor:     控制batchsize的大小
    input:      dir_path ::= 图片存储地址
                imgorg_size ::= [img_w, img_h, x_offset, y_offset],
                            img_w, img_h ::= 进行关键点检测的图片的尺寸
                            x_offset, y_offset ::= 进行关键点检测的图片, 从原始大图中crop的坐标位移
                file_dst ::= 存储检测的关键点， 格式如下面介绍


    return:     一个dict
                key: 图像名称
                value: 一个list， 数据维度 1*11，为关键点+关键点得分
    """
    use_gpu = 1

    modelpre=torch.load('../model/kp_bgr_8x8_epoch_100.pkl')
    #modelpre = torch.load('../model/kp_bgr_8x8_epoch_100.pkl', map_location=lambda storage, loc: storage)
    if use_gpu:
        modelpre.cuda()
    else:
        device = torch.device('cpu')
        modelpre.to(device)
    sd_pre=modelpre.state_dict()
  
    model=KPNet_FullConv_RGB_8x8()
    if use_gpu: 
        model.cuda()
    else:
        model.to(device)
    sd=model.state_dict()
   # 
    for key in sd:
        sd[key]=sd_pre[key]
    model.load_state_dict(sd)
    model.eval()

    w,h=img.size
    img=img.crop((int(w*0.2),int(h*0.2),int(w*0.8),int(h*0.8))).resize((128,128))
    scalew = w*0.6/128
    scaleh = h*0.6/128

    input=np.array(img,dtype=np.float32).swapaxes(1,2).swapaxes(0,1).reshape((1,3,128,128))-128
    input=np.flip(input,axis=1).copy()
    if use_gpu:
        input=torch.Tensor(input).float().cuda()
    else:
        input=torch.Tensor(input).float().to(device)

    # imgarr=np.zeros((1,1,96,96))

    # img=Image.open(img_path)
    # img = img.resize([160, 160])
    # if np.array(img).shape[0] == 160:  
    #     img = img.crop([32,32,128,128])    # 缩放到96 * 96进行关键点检测   crop((40,40,160,160))
    # img=np.array(img,dtype=np.float32).reshape(1, 1,96,96)
    # imgarr=img-128
    #input = torch.Tensor(imgarr).float().cuda()
    
    output = model(input)
    
    kp = output2KP(output)

    
    cont_dict = {}
    xy = kp[0][5]
    kp_list = []
    for x,y in kp[0][:5]:
        x = x * scalew + 0.2*w
        y = y * scaleh + 0.2*h
        kp_list.append(x)
        kp_list.append(y)
    kp_list.append(xy)

    cont_dict['keypoints'] = kp_list

    return kp_list, cont_dict

def toRed(img, place):
    value = np.zeros((9,9,3))
    value[:, :, 0] = np.ones((9,9)) * 255
    try:
        img[place[0]-4:place[0]+5, place[1]-4:place[1]+5, :] = value
    except:
        pass
   


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description = 'Get keypoints')
    parser.add_argument('-d', '--img_path', type = str, default = '')
    parser.add_argument('--use_gpu',  type = str, default = True)

    args = parser.parse_args()
    print(" ------ Get Face Keypoints ------")

    img_path = args.img_path
    keypoint_path = os.path.join('../results', os.path.basename(img_path).replace('.jpg', '.json'))

    img = Image.open(img_path)

    kp_list, cont_dict = get_keypoint(img, imgorg_size=[96, 96, 32, 32], factor = 100)

    with open(keypoint_path, 'w') as fd:
        json.dump(cont_dict, fd)

    img = np.array(img)    
    for i in range(5):
        place = [int(kp_list[2*i + 1]), int(kp_list[2*i])]
        toRed(img, place)
    newImg = Image.fromarray(img)

    newImg.save(os.path.join('../results', os.path.basename(img_path).replace('.jpg', '_kp.jpg')))
    print(" ------ Finished ------")


    




