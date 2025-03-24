import torch
import torch.nn as nn
from torch.nn import functional as F

"""
把ResNet18的残差卷积单元作为一个Block，这里分为两种：一种是CommonBlock，另一种是SpecialBlock，最后由ResNet18统筹调度
其中SpecialBlock负责完成ResNet18中带有虚线（升维channel增加和下采样操作h和w减少）的Block操作
其中CommonBlock负责完成ResNet18中带有实线的直接相连相加的Block操作
注意ResNet18中所有非shortcut部分的卷积kernel_size=3， padding=1，仅仅in_channel, out_channel, stride的不同 
注意ResNet18中所有shortcut部分的卷积kernel_size=1， padding=0，仅仅in_channel, out_channel, stride的不同
"""
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
        self.fc = nn.Sequential(  # 最后用于分类的全连接层，根据需要灵活变化
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, classes_num),
            nn.ReLU(inplace=True)
        )
        self.cbam1 = CBAMLayer(channel=64)
        self.cbam2 = CBAMLayer(channel=64)
        self.cbam3 = CBAMLayer(channel=128)
        self.cbam4 = CBAMLayer(channel=256)
        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes_num)

    def forward(self, x):
        x_rgb = self.prepare(x)  # 三通道 64
        x = self.layer1(x_rgb)
        x = self.cbam2(x) * x + x
        x = self.layer2(x)
        x = self.cbam3(x) * x + x
        x = self.layer3(x)
        x = self.cbam4(x) * x + x
        x = self.layer4(x)
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
