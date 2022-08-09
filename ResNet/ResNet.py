'''
Author:leo
Date:22/08/07
Version:1.0
'''
import torch
import torch.nn as nn

class BasicBlock(nn.Module):     # for 18 layers and 34 layers
    expansion=1                  # 

    def __init__(self,in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super().__init__()
        # 使用BN时，bias=False，BN放在conv层和relu层之间(似乎可以调整，效果还不太一样)
        self.conv1=nn.Conv2d(in_channels=in_channel,out_channels=out_channel,
                kernel_size=3,stride=stride,padding=1,bias=False)  #对不同的结构，stride不同，输出的图像size不变
        self.bn=nn.BatchNorm2d(out_channel)
        self.relu=nn.ReLU()

        self.conv2=nn.Conv2d(in_channels=out_channel,out_channels=out_channel,
                kernel_size=3,stride=1,padding=1,bias=False)
        

        self.downsample=downsample

    def forward(self,x):
        identity=x
        if self.downsample is not None:  #size减半，channel翻倍
            identity=self.downsample(x)

        out=self.conv1(x)
        out=self.bn(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn(out)
        
        out+=identity           # 不是做concatenate
        out=self.relu(out)      # 相加后再做relu

        return out

class Bottleneck(nn.Module):    # for 50 101 152 layers
    expansion=4                 # Block最后一层out_channel为之前的4倍

    def __init__(self,in_channel, out_channel, stride=1, downsample=None):
        super().__init__()

        self.conv1=nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,bias=False)
        self.bn=nn.BatchNorm2d(out_channel)
        
        self.conv2=nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3
                    ,stride=stride,padding=1,bias=False)

        self.conv3=nn.Conv2d(in_channels=out_channel,out_channels=out_channel*expansion,kernel_size=1,stride=1,bias=False)
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample

    def forward(self,x):
        identity=x
        if self.downsample is not None:  #size减半，channel翻倍
            identity=self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn(out)

        out += identity
        out = self.relu(out)

        return out



class ResNet(nn.Module):
    def __init__(self,block,blocks_num,num_classes=1000,include_top=True):
        super().__init__()
        self.include_top=include_top
        self.in_channel=64

        self.conv1=nn.Conv2d(3,out_channels=self.in_channel,kernel_size=7
                    ,stride=2,padding=3,bias=False)  # rgb image
        self.bn1=nn.BatchNorm2d(self.in_channel)
        self.relu=nn.ReLU(inplace=True)

        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def _make_layer(self,block,channel,block_num,stride=1): 
        downsample=None
        if stride!=1 or self.in_channel!=channel*block.expansion:  # for 50 101 152
            #虚线连接部分（f(x)和x的channel不同）
            downsample=nn.Sequential(nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False)
                                ,nn.BatchNorm2d(channel * block.expansion))
            
        layers=[]
        layers.append(block(self.in_channel,channel,downsample=downsample
                        ,stride=stride)) 
                        # for 50 101 152 in_channel != channel  channel一方面用来判断哪种结构一方面最后一层in_channel=channel*expansion

        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                    channel))

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

def resnet34(num_classes=1000,include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet50(num_classes=1000,include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet101(num_classes=1000,include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
