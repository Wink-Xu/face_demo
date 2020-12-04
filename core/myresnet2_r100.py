from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, \
    MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
from collections import namedtuple
import math
import pdb
import torch
import torch.nn as nn
import numpy as np
import torchvision.models.resnet

##################################  Original Arcface Model #############################################################

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth, momentum=0.9, eps=2e-5))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel,momentum=0.9,eps=2e-5),
            Conv2d(in_channel, depth, (3, 3),(1, 1), 1, bias=False),
            BatchNorm2d(depth,momentum=0.9,eps=2e-5),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth,momentum=0.9,eps=2e-5))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class Attention(Module):
    def __init__(self, innum, outnum):
        super(Attention, self).__init__()
        self.att = Sequential(
                                   Conv2d(innum, outnum, (3, 3), 2, 1, bias=False),
                                   BatchNorm2d(outnum,momentum=0.9,eps=2e-5),
                                   PReLU(outnum),
                                   Conv2d(outnum, outnum, (3, 3), 2, 1, bias=False),
                                   BatchNorm2d(outnum, momentum=0.9, eps=2e-5),
                                   PReLU(outnum),
                                   Conv2d(outnum, outnum, (3, 3), 2, 1, bias=False),
                                   BatchNorm2d(outnum, momentum=0.9, eps=2e-5),
                                   PReLU(outnum),
                                   AdaptiveAvgPool2d(1),
                                   Conv2d(outnum, outnum, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(outnum, momentum=0.9, eps=2e-5),
                                   Sigmoid()
                              )
    def forward(self, x):
        x = self.att(x)
        return x



def make_layers(block,innum,outnum,num,stride=2):
    downsample = None
    layers = []
    layers.append(block(innum, outnum, stride))
    for i in range(1,num):
        layers.append(block(outnum, outnum,1))
    return nn.Sequential(*layers)

class Backbone(Module):
    def __init__(self, classnum):
        super(Backbone, self).__init__()

        unit_module = bottleneck_IR

        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64,momentum=0.9,eps=2e-5),
                                      PReLU(64))
        self.output_layer = Sequential(BatchNorm2d(512,momentum=0.9,eps=2e-5),
                                       Dropout(0.4),
                                       Flatten(),
                                       Linear(512 * 7 * 7, 512),
                                       BatchNorm1d(512,momentum=0.9,eps=2e-5))

        self.layer1=make_layers(unit_module,64,64,3)
        self.layer2_1=make_layers(unit_module,64,128,6)
        self.layer2_2=make_layers(unit_module,128,128,7,stride=1)
        self.layer3_1=make_layers(unit_module,128,256,10)
        self.layer3_2=make_layers(unit_module,256,256,10,stride=1)
        self.layer3_3=make_layers(unit_module,256,256,10,stride=1)
        self.layer4=make_layers(unit_module,256,512,3)

        self.ac_fc=Arcface(embedding_size=512,classnum=classnum)

        self.att1=Sequential(
                                   Conv2d(3, 64, (3, 3), 2, 1, bias=False),
                                   BatchNorm2d(64,momentum=0.9,eps=2e-5),
                                   PReLU(64),
                                   Conv2d(64, 64, (3, 3), 2, 1, bias=False),
                                   BatchNorm2d(64, momentum=0.9, eps=2e-5),
                                   PReLU(64),
                                   Conv2d(64, 64, (3, 3), 2, 1, bias=False),
                                   BatchNorm2d(64, momentum=0.9, eps=2e-5),
                                   PReLU(64),
                                   Conv2d(64, 64, (3, 3), 2, 1, bias=False),
                                   BatchNorm2d(64, momentum=0.9, eps=2e-5),
                                   PReLU(64),
                                   AdaptiveAvgPool2d(1),
                                   Conv2d(64, 64, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(64, momentum=0.9, eps=2e-5),
                                   Sigmoid()
                              )
        self.att2_1=Sequential(
                                   Conv2d(64, 128, (3, 3), 2, 1, bias=False),
                                   BatchNorm2d(128,momentum=0.9,eps=2e-5),
                                   PReLU(128),
                                   Conv2d(128, 128, (3, 3), 2, 1, bias=False),
                                   BatchNorm2d(128, momentum=0.9, eps=2e-5),
                                   PReLU(128),
                                   Conv2d(128, 128, (3, 3), 2, 1, bias=False),
                                   BatchNorm2d(128, momentum=0.9, eps=2e-5),
                                   PReLU(128),
                                   AdaptiveAvgPool2d(1),
                                   Conv2d(128, 128, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(128, momentum=0.9, eps=2e-5),
                                   Sigmoid()
                              )
        self.att2_2=Sequential(
                                   Conv2d(128, 128, (3, 3), 2, 1, bias=False),
                                   BatchNorm2d(128,momentum=0.9,eps=2e-5),
                                   PReLU(128),
                                   Conv2d(128, 128, (3, 3), 2, 1, bias=False),
                                   BatchNorm2d(128, momentum=0.9, eps=2e-5),
                                   PReLU(128),
                                   Conv2d(128, 128, (3, 3), 1, 1, bias=False),
                                   BatchNorm2d(128, momentum=0.9, eps=2e-5),
                                   PReLU(128),
                                   AdaptiveAvgPool2d(1),
                                   Conv2d(128, 128, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(128, momentum=0.9, eps=2e-5),
                                   Sigmoid()
                              )
        self.att3_1=Sequential(
                                   Conv2d(128, 256, (3, 3), 2, 1, bias=False),
                                   BatchNorm2d(256,momentum=0.9,eps=2e-5),
                                   PReLU(256),
                                   Conv2d(256, 256, (3, 3), 2, 1, bias=False),
                                   BatchNorm2d(256, momentum=0.9, eps=2e-5),
                                   PReLU(256),
                                   Conv2d(256, 256, (3, 3), 1, 1, bias=False),
                                   BatchNorm2d(256, momentum=0.9, eps=2e-5),
                                   PReLU(256),
                                   AdaptiveAvgPool2d(1),
                                   Conv2d(256, 256, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(256, momentum=0.9, eps=2e-5),
                                   Sigmoid()
                              )

        self.att3_2=Sequential(
                                   Conv2d(256, 256, (3, 3), 2, 1, bias=False),
                                   BatchNorm2d(256,momentum=0.9,eps=2e-5),
                                   PReLU(256),
                                   Conv2d(256, 256, (3, 3), 1, 1, bias=False),
                                   BatchNorm2d(256, momentum=0.9, eps=2e-5),
                                   PReLU(256),
                                   Conv2d(256, 256, (3, 3), 1, 1, bias=False),
                                   BatchNorm2d(256, momentum=0.9, eps=2e-5),
                                   PReLU(256),
                                   AdaptiveAvgPool2d(1),
                                   Conv2d(256, 256, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(256, momentum=0.9, eps=2e-5),
                                   Sigmoid()
                              )

        self.att3_3=Sequential(
                                   Conv2d(256, 256, (3, 3), 2, 1, bias=False),
                                   BatchNorm2d(256,momentum=0.9,eps=2e-5),
                                   PReLU(256),
                                   Conv2d(256, 256, (3, 3), 1, 1, bias=False),
                                   BatchNorm2d(256, momentum=0.9, eps=2e-5),
                                   PReLU(256),
                                   Conv2d(256, 256, (3, 3), 1, 1, bias=False),
                                   BatchNorm2d(256, momentum=0.9, eps=2e-5),
                                   PReLU(256),
                                   AdaptiveAvgPool2d(1),
                                   Conv2d(256, 256, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(256, momentum=0.9, eps=2e-5),
                                   Sigmoid()
                              )

        self.att4=Sequential(
                                   Conv2d(256, 512, (3, 3), 2, 1, bias=False),
                                   BatchNorm2d(512,momentum=0.9,eps=2e-5),
                                   PReLU(512),
                                   Conv2d(512, 512, (3, 3), 1, 1, bias=False),
                                   BatchNorm2d(512, momentum=0.9, eps=2e-5),
                                   PReLU(512),
                                   Conv2d(512, 512, (3, 3), 1, 1, bias=False),
                                   BatchNorm2d(512, momentum=0.9, eps=2e-5),
                                   PReLU(512),
                                   AdaptiveAvgPool2d(1),
                                   Conv2d(512, 512, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(512, momentum=0.9, eps=2e-5),
                                   Sigmoid()
                              )




        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(3. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.out_features
                m.weight.data.normal_(0, math.sqrt(3. / n))


    def forward(self, input,labels):
        x = self.input_layer(input)
        w1 = self.att1(input)
        x = self.layer1(x)*w1

        w2_1 = self.att2_1(x)
        x = self.layer2_1(x)*w2_1

        w2_2 = self.att2_2(x)
        x = self.layer2_2(x)*w2_2

        w3_1 = self.att3_1(x)
        x = self.layer3_1(x)*w3_1

        w3_2 = self.att3_2(x)
        x = self.layer3_2(x)*w3_2

        w3_3 = self.att3_3(x)
        x = self.layer3_3(x)*w3_3

        w4 = self.att4(x)
        x = self.layer4(x)*w4

        # mean1=torch.mean(torch.sort(torch.abs(w1[0]-w1[-1]).view(-1))[0][-16:])
        # mean2=torch.mean(torch.sort(torch.abs(w2[0]-w2[-1]).view(-1))[0][-32:])
        # mean3=torch.mean(torch.sort(torch.abs(w3[0]-w3[-1]).view(-1))[0][-64:])
        # mean4=torch.mean(torch.sort(torch.abs(w4[0]-w4[-1]).view(-1))[0][-128:])
        #
        # print(mean1.item(),mean2.item(),mean3.item(),mean4.item())


        v = self.output_layer(x)
        y = self.ac_fc(v,labels)
        return y,v


##################################  Arcface head #############################################################

class Arcface(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, classnum=51332, s=64., m=0.5):
        super(Arcface, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        # initial kernel
        # self.kernel.data.uniform_(-1, 1)
        self.kernel.data.normal_(0, math.sqrt(3. / classnum))
        self.m = m  # the margin value, default is 0.5
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)

    def forward(self, embbedings, label):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel, axis=0)
        embbedings=l2_norm(embbedings)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings, kernel_norm)
        #         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm)  # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0  # a little bit hacky way to prevent in_place operation on cos_theta
        # idx_ = torch.arange(0, nB, dtype=torch.long)
        # output[idx_, label] = cos_theta_m[idx_, label]

        output[torch.Tensor([[i] for i in range(nB)]).to(torch.long), label] = cos_theta_m[
            torch.Tensor([[i] for i in range(nB)]).to(torch.long), label]

        output *= self.s  # scale up in order to make softmax work, first introduced in normface

        return output


if __name__=='__main__':
    import numpy as np
    import random
    # a=torch.rand((5,3))
    # print(torch.mean(a))
    # b=a.t()
    # # b=torch.rand((7,5))
    # label=torch.Tensor([1,2,3,3,4]).long()
    # label=torch.unique(label)
    # print(label)
    # print(a)
    # print(a[label,:])
    # print(b[:,label])
    # a.requires_grad=True
    # b.requires_grad=True
    # # b=a.t()
    # # print(b.requires_grad)
    # c=torch.mm(b,a)
    # d=torch.sum(c)
    # d.backward()
    # print(a.grad)
    # print(b.grad)

    a=torch.Tensor(np.random.random(size=(512, 20))-0.5)
    a=l2_norm(a,0)
    b=a.t()
    print(torch.mm(b,a))



