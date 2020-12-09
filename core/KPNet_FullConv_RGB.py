import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class KPNet_FullConv_RGB(nn.Module):
    def __init__(self):
        super(KPNet_FullConv_RGB,self).__init__()
        self.head=nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.output=nn.Sequential(
            nn.Conv2d(512, 15, kernel_size=1),
            nn.BatchNorm2d(15),
            nn.Sigmoid()
        )

    def forward(self, x):
        x=self.head(x)
        y=self.output(x)
        return y


class KPNet_FullConv_RGB_8x8(nn.Module):
    def __init__(self):
        super(KPNet_FullConv_RGB_8x8,self).__init__()
        self.head=nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 256, 4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.output=nn.Sequential(
            nn.Conv2d(512, 15, kernel_size=1),
            nn.BatchNorm2d(15),
            nn.Sigmoid()
        )

    def forward(self, x):
 #       import pdb; pdb.set_trace()
        x=self.head(x)
        y=self.output(x)
        return y




class KPNet_FullConv_Gray_8x8(nn.Module):
    def __init__(self):
        super(KPNet_FullConv_Gray_8x8,self).__init__()
        self.head=nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 256, 4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.output=nn.Sequential(
            nn.Conv2d(512, 15, kernel_size=1),
            nn.BatchNorm2d(15),
            nn.Sigmoid()
        )

    def forward(self, x):
        x=self.head(x)
        y=self.output(x)
        return y

if __name__=='__main__':
    seq=nn.Sequential()
    x=torch.Tensor(size=(1,3,16,16))
    y=seq(x)
    print(y.size())
