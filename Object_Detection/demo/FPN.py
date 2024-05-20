import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,inplanes,planes,downsample=None,stride = 1):
        super(Bottleneck,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inplanes,planes,1,bias = False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace = True),
            nn.Conv2d(planes,planes,3,stride,1,bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes,planes*self.expansion,1,bias=False),
            nn.BatchNorm2d(planes*self.expansion),
        )
        self.relu = nn.ReLU(inplace=True)
        self.inception = downsample

    def forward(self,x):
        out = self.block(x)
        if self.inception:
            indef = self.inception(x)
        else:
            indef = x
        out = out + indef
        out = self.relu(out)
        return out



class FPN(nn.Module):
    def __init__(self,layers):
        super(FPN,self).__init__()
        self.inplanes = 64
        self.conv = nn.Conv2d(3,64,7,2,3,bias = False)
        self.maxpooling = nn.MaxPool2d(3,2,1)
        self.relu = nn.ReLU(inplace = True)
        self.batchnorm = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64,layers[0])
        self.layer2 = self._make_layer(128,layers[1],2)
        self.layer3 = self._make_layer(256,layers[2],2)
        self.layer4 = self._make_layer(512,layers[3],2)
        self.toplayer = nn.Conv2d(2048,256,1,1,0)
        self.smooth1 = nn.Conv2d(256,256,3,1,1)
        self.smooth2 = nn.Conv2d(256,256,3,1,1)
        self.smooth3 = nn.Conv2d(256,256,3,1,1)
        self.latlayer1 = nn.Conv2d(256,256,1,1,0)
        self.latlayer2 = nn.Conv2d(512,256,1,1,0)
        self.latlayer3 = nn.Conv2d(1024,256,1,1,0)

    def _make_layer(self,planes,num,stride = 1):
        downsample = None
        layers =[]
        if stride !=1 or self.inplanes != planes*Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,planes*Bottleneck.expansion,1,stride,bias = False),
                nn.BatchNorm2d(planes*Bottleneck.expansion)
            )
        layers.append(Bottleneck(self.inplanes,planes,downsample,stride))
        # 只有第一个块输入维度与输出维度发生变化，所以需要downsample
        self.inplanes = planes*Bottleneck.expansion
        #其余块输入与输出维度相同
        for i in range(1,num):
            layers.append(Bottleneck(self.inplanes,planes))
        return nn.Sequential(*layers)

    def _upsample_add(self,x,y):
        _,_,H,W = y.shape
        return F.upsample(x,size = (H,W),mode = 'bilinear')+y

    def forward(self,x):
        out = self.conv(x)
        out = self.batchnorm(out)
        out = self.relu(out)
        out = self.maxpooling(out)
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        p4 = self.toplayer(out4)
        p3 = self._upsample_add(p4,self.latlayer3(out3))
        p2 = self._upsample_add(p3,self.latlayer2(out2))
        p1 = self._upsample_add(p2,self.latlayer1(out1))

        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth1(p1)
        return p4,p3,p2,p1


if __name__ == '__main__':
    fpn = FPN([3,4,6,3]).cuda()
    pdb.set_trace()