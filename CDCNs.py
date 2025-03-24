'''
Code of 'Searching Central Difference Convolutional Networks for Face Anti-Spoofing' 
By Zitong Yu & Zhuo Su, 2019

If you use the code, please cite:
@inproceedings{yu2020searching,
    title={Searching Central Difference Convolutional Networks for Face Anti-Spoofing},
    author={Yu, Zitong and Zhao, Chenxu and Wang, Zezheng and Qin, Yunxiao and Su, Zhuo and Li, Xiaobai and Zhou, Feng and Zhao, Guoying},
    booktitle= {CVPR},
    year = {2020}
}

Only for research purpose, and commercial use is not allowed.

MIT License
Copyright (c) 2020 
'''

import math

import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import Parameter
import pdb
import numpy as np


########################   Centeral-difference (second order, with 9 parameters and a const theta for 3x3 kernel) 2D Convolution   ##############################
## | a1 a2 a3 |   | w1 w2 w3 |
## | a4 a5 a6 | * | w4 w5 w6 | --> output = \sum_{i=1}^{9}(ai * wi) - \sum_{i=1}^{9}wi * a5 --> Conv2d (k=3) - Conv2d (k=1)
## | a7 a8 a9 |   | w7 w8 w9 |
##
##   --> output = 
## | a1 a2 a3 |   |  w1  w2  w3 |     
## | a4 a5 a6 | * |  w4  w5  w6 |  -  | a | * | w\_sum |     (kernel_size=1x1, padding=0)
## | a7 a8 a9 |   |  w7  w8  w9 |     

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)

            return out_normal - self.theta * out_diff


class SpatialAttention(nn.Module):
    def __init__(self, kernel=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x)

class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x
class CDCN(nn.Module):

    def __init__(self, basic_conv=Conv2d_cd, theta=0.7):
        super(CDCN, self).__init__()

        self.conv1 = nn.Sequential(
            basic_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.Block1 = nn.Sequential(
            basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

        )

        self.Block2 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.Block3 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.lastconv1 = nn.Sequential(
            basic_conv(128 * 3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.lastconv2 = nn.Sequential(
            basic_conv(128, 64, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.lastconv3 = nn.Sequential(
            basic_conv(64, 1, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.ReLU(),
        )

        self.downsample32x32 = nn.Upsample(size=(32, 32), mode='bilinear')
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.linear = nn.Linear(16*16*16, 2)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.cbma1 = CBAMLayer(128)

    def forward(self, x):  # x [3, 256, 256]

        x_input = x
        x = self.conv1(x)

        x_Block1 = self.Block1(x)  # x [128, 128, 128]
        x_Block1_32x32 = self.downsample32x32(x_Block1)  # x [128, 32, 32]

        x_Block2 = self.Block2(x_Block1)  # x [128, 64, 64]
        x_Block2_32x32 = self.downsample32x32(x_Block2)  # x [128, 32, 32]

        x_Block3 = self.Block3(x_Block2)  # x [128, 32, 32]
        x_Block3_32x32 = self.downsample32x32(x_Block3)  # x [128, 32, 32]

        x_concat = torch.cat((x_Block1_32x32, x_Block2_32x32, x_Block3_32x32), dim=1)  # x [128*3, 32, 32]

        # pdb.set_trace()

        x = self.lastconv1(x_concat)  # x [128, 32, 32]
        x = self.lastconv2(x)  # x [64, 32, 32]
        x = self.lastconv3(x)  # x [1, 32, 32]

        x = self.features(x)
        x = x.view(x.size(0), -1)
        # x = self.dropout(F.relu(self.fc1(x)))
        # x = self.dropout(F.relu(self.fc2(x)))
        # x = F.relu(self.fc3(x))
        x = self.linear(x)


        # map_x = x.squeeze(1)

        # return map_x, x_concat, x_Block1, x_Block2, x_Block3, x_input
        return x
class EfficientChannelAttention(nn.Module):
    def __init__(self, C, gamma=2, b=1):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(C, 2) + b) / gamma))
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
class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = self.max_pool(x)
        max_out = self.mlp(max_pool)
        avg_pool = self.avg_pool(x)
        avg_out = self.mlp(avg_pool)
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        conv = self.conv(torch.cat([max_out, avg_out], dim=1))
        spatial_out = self.sigmoid(conv)
        x = spatial_out * x
        return x

class Block1(nn.Module):
    def __init__(self, basic_conv=Conv2d_cd, theta=0.7):
        super(Block1, self).__init__()
        self.basic_conv1 = basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta)
        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

        self.basic_conv2 = basic_conv(128, int(128 * 1.6), kernel_size=3, stride=1, padding=1, bias=False, theta=theta)
        self.bn2 = nn.BatchNorm2d(int(128 * 1.6))

        self.basic_conv3 = basic_conv(int(128 * 1.6), 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta)
        self.bn3 = nn.BatchNorm2d(128)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        identify = self.basic_conv1(x)
        x = self.relu(self.bn(self.basic_conv1(x)))
        x = self.relu(self.bn2(self.basic_conv2(x)))
        x = self.bn3(self.basic_conv3(x))
        x += identify

        x = self.relu(x)
        x = self.maxpool(x)
        return x

class Block2(nn.Module):
    def __init__(self, basic_conv=Conv2d_cd, theta=0.7):
        super(Block2, self).__init__()
        self.basic_conv1 = basic_conv(128, int(128 * 1.2), kernel_size=3, stride=1, padding=1, bias=False, theta=theta)
        self.bn = nn.BatchNorm2d(int(128 * 1.2))
        self.relu = nn.ReLU()

        self.basic_conv2 = basic_conv(int(128 * 1.2), 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta)
        self.bn2 = nn.BatchNorm2d(128)

        self.basic_conv3 = basic_conv(128, int(128 * 1.4), kernel_size=3, stride=1, padding=1, bias=False, theta=theta)
        self.bn3 = nn.BatchNorm2d(int(128 * 1.4))

        self.basic_conv4 = basic_conv(int(128 * 1.4), 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta)
        self.bn4 = nn.BatchNorm2d(128)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        identify = x
        x = self.relu(self.bn(self.basic_conv1(x)))
        x = self.bn2(self.basic_conv2(x))
        x += identify

        identify1 = x
        x = self.relu(x)
        x = self.relu(self.bn3(self.basic_conv3(x)))
        x = self.bn4(self.basic_conv4(x))

        x += identify1
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class Block3(nn.Module):
    def __init__(self, basic_conv=Conv2d_cd, theta=0.7):
        super(Block3, self).__init__()
        self.basic_conv1 = basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

        self.basic_conv2 = basic_conv(128, int(128 * 1.2), kernel_size=3, stride=1, padding=1, bias=False, theta=theta)
        self.bn2 = nn.BatchNorm2d(int(128 * 1.2))

        self.basic_conv3 = basic_conv(int(128 * 1.2), 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta)
        self.bn3 = nn.BatchNorm2d(128)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        identify = x
        x = self.relu(self.bn1(self.basic_conv1(x)))
        x = self.bn2(self.basic_conv2(x))
        x = self.relu(x)
        x = self.bn3(self.basic_conv3(x))
        x += identify
        x = self.relu(x)
        x = self.maxpool(x)
        return x
class CDCNpp(nn.Module):

    def __init__(self, basic_conv=Conv2d_cd, theta=0.7):
        super(CDCNpp, self).__init__()

        self.conv1 = nn.Sequential(
            basic_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )

        # self.Block1 = nn.Sequential(
        #     basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #
        #     basic_conv(128, int(128 * 1.6), kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
        #     nn.BatchNorm2d(int(128 * 1.6)),
        #     nn.ReLU(),
        #     basic_conv(int(128 * 1.6), 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #
        # )
        #
        # self.Block2 = nn.Sequential(
        #     basic_conv(128, int(128 * 1.2), kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
        #     nn.BatchNorm2d(int(128 * 1.2)),
        #     nn.ReLU(),
        #     basic_conv(int(128 * 1.2), 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     basic_conv(128, int(128 * 1.4), kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
        #     nn.BatchNorm2d(int(128 * 1.4)),
        #     nn.ReLU(),
        #     basic_conv(int(128 * 1.4), 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        # )
        #
        # self.Block3 = nn.Sequential(
        #     basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     basic_conv(128, int(128 * 1.2), kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
        #     nn.BatchNorm2d(int(128 * 1.2)),
        #     nn.ReLU(),
        #     basic_conv(int(128 * 1.2), 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        # )


        self.Block1 = Block1()
        self.Block2 = Block2()
        self.Block3 = Block3()

        # Original

        self.lastconv1 = nn.Sequential(
            basic_conv(128 * 3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 1, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.ReLU(),
        )

        self.sa1 = SpatialAttention(kernel=7)
        self.sa2 = SpatialAttention(kernel=5)
        self.sa3 = SpatialAttention(kernel=3)
        self.downsample32x32 = nn.Upsample(size=(32, 32), mode='bilinear')
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.linear = nn.Linear(4096, 2)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.se = EfficientChannelAttention(384)
        self.cbam = CBAMLayer(128)
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):  # x [3, 256, 256]
        x = self.conv1(x)

        x_Block1 = self.Block1(x)
        # attention1 = self.sa1(x_Block1)
        attention1 = self.cbam(x_Block1)
        x_Block1_SA = attention1 * x_Block1 + x_Block1
        x_Block1_32x32 = self.downsample32x32(x_Block1_SA)

        x_Block2 = self.Block2(x_Block1)
        # attention2 = self.sa2(x_Block2)
        attention2 = self.cbam(x_Block2)
        x_Block2_SA = attention2 * x_Block2 + x_Block2
        x_Block2_32x32 = self.downsample32x32(x_Block2_SA)

        x_Block3 = self.Block3(x_Block2)
        # attention3 = self.sa3(x_Block3)
        attention3 = self.cbam(x_Block3)
        x_Block3_SA = attention3 * x_Block3 + x_Block3
        x_Block3_32x32 = self.downsample32x32(x_Block3_SA)

        x_Block1_32x32 = self.se(x_Block1_32x32) * x_Block1_32x32
        x_Block2_32x32 = self.se(x_Block2_32x32) * x_Block2_32x32
        x_Block3_32x32 = self.se(x_Block3_32x32) * x_Block3_32x32

        x_concat = torch.cat((x_Block1_32x32, x_Block2_32x32, x_Block3_32x32), dim=1)
        # map_x = self.se(x_concat) * x_concat
        map_x = self.lastconv1(x_concat)
        #
        # map_x = map_x.squeeze(1)
        # x = self.features(map_x)
        x = map_x.view(map_x.size(0), -1)
        # x = self.linear(x)
        #
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))


        return x, map_x
if __name__ == "__main__":
    # model = CDCN().to(device="cuda")
    model = CDCNpp().to(device='cuda')
    x = torch.zeros((1, 3, 64, 64), device="cuda")
    output = model(x)
    print(output)