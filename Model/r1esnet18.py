import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary
from math import log
from CDCNs import CBAMLayer

"""
把ResNet18的残差卷积单元作为一个Block，这里分为两种：一种是CommonBlock，另一种是SpecialBlock，最后由ResNet18统筹调度
其中SpecialBlock负责完成ResNet18中带有虚线（升维channel增加和下采样操作h和w减少）的Block操作
其中CommonBlock负责完成ResNet18中带有实线的直接相连相加的Block操作
注意ResNet18中所有非shortcut部分的卷积kernel_size=3， padding=1，仅仅in_channel, out_channel, stride的不同 
注意ResNet18中所有shortcut部分的卷积kernel_size=1， padding=0，仅仅in_channel, out_channel, stride的不同
"""
class EfficientChannelAttention(nn.Module):
    def __init__(self, C, gamma=2, b=1):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((log(C, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class CommonBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):  # 普通Block简单完成两次卷积操作
        super(CommonBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = x  # 普通Block的shortcut为直连，不需要升维下采样

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)  # 完成一次卷积
        x = self.bn2(self.conv2(x))  # 第二次卷积不加relu激活函数

        x += identity  # 两路相加
        return F.relu(x, inplace=True)  # 添加激活函数输出


class SpecialBlock(nn.Module):  # 特殊Block完成两次卷积操作，以及一次升维下采样
    def __init__(self, in_channel, out_channel, stride):  # 注意这里的stride传入一个数组，shortcut和残差部分stride不同
        super(SpecialBlock, self).__init__()
        self.change_channel = nn.Sequential(  # 负责升维下采样的卷积网络change_channel
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride[0], padding=0, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride[1], padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = self.change_channel(x)  # 调用change_channel对输入修改，为后面相加做变换准备

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))  # 完成残差部分的卷积

        x += identity
        return F.relu(x, inplace=True)  # 输出卷积单元


class ResNet18(nn.Module):
    def __init__(self, classes_num=2):
        super(ResNet18, self).__init__()
        self.prepare = nn.Sequential(  # 所有的ResNet共有的预处理==》[batch, 64, 56, 56]
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = nn.Sequential(  # layer1有点特别，由于输入输出的channel均是64，故两个CommonBlock
            nn.Conv2d(64, 64, 3, 1, 1),
            CommonBlock(64, 64, 1),
            CommonBlock(64, 64, 1)
        )
        self.layer2 = nn.Sequential(  # layer234类似，由于输入输出的channel不同，故一个SpecialBlock，一个CommonBlock
            SpecialBlock(64, 128, [2, 1]),
            CommonBlock(128, 128, 1)
        )
        self.layer3 = nn.Sequential(
            SpecialBlock(128, 256, [2, 1]),
            CommonBlock(256, 256, 1)
        )
        self.layer4 = nn.Sequential(
            SpecialBlock(256, 512, [2, 1]),
            CommonBlock(512, 512, 1)
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(4, 4))  # 卷积结束，通过一个自适应均值池化==》 [batch, 512, 1, 1]
        self.fc = nn.Sequential(  # 最后用于分类的全连接层，根据需要灵活变化
            nn.Dropout(p=0.5),
            nn.Linear(768, 512),
            nn.ReLU(inplace=True),  # 这个使用CIFAR10数据集，定为10分类
        )
        self.cbam1 = CBAMLayer(channel=64)
        self.cbam2 = CBAMLayer(channel=64)
        self.cbam3 = CBAMLayer(channel=128)
        self.cbam4 = CBAMLayer(channel=256)
        # self.cbam5 = CBAMLayer(channel=512)
        self.att = EfficientChannelAttention(960)
        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(960, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes_num)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x_rgb):
        x_rgb = self.prepare(x_rgb)  # 三通道 64
        x = self.layer1(x_rgb)
        x = self.cbam2(x) * x + x

        x1 = self.avg_pool(x)

        x = self.layer2(x)
        x = self.cbam3(x) * x + x
        x2 = self.avg_pool(x)

        x = self.layer3(x)
        x = self.cbam4(x) * x + x
        x3 = self.avg_pool(x)

        x = self.layer4(x)
        x4 = self.avg_pool(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = x * self.att(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        return x


if __name__ == "__main__":
    model = ResNet18(classes_num=2).to(device="cuda")
    x = torch.randn(1, 3, 64, 64).to(device='cuda')
    # summary(model, (3, 64, 64))
    output = model(x)
    print(output)
