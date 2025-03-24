from Model.resnet_1018 import *


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


class CommonBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride, basic_conv=Conv2d_cd):
        super(CommonBlock, self).__init__()
        self.conv1 = basic_conv(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = basic_conv(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = x

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))

        x += identity
        return F.relu(x, inplace=True)


class SpecialBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride, basic_conv=Conv2d_cd):
        super(SpecialBlock, self).__init__()
        self.change_channel = nn.Sequential(
            basic_conv(in_channel, out_channel, kernel_size=1, stride=stride[0], padding=0, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.conv1 = basic_conv(in_channel, out_channel, kernel_size=3, stride=stride[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = basic_conv(out_channel, out_channel, kernel_size=3, stride=stride[1], padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = self.change_channel(x)

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))

        x += identity
        return F.relu(x, inplace=True)


class ResNet34(nn.Module):
    def __init__(self, classes_num=2, basic_conv=Conv2d_cd):
        super(ResNet34, self).__init__()
        self.prepare = nn.Sequential(
            basic_conv(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = nn.Sequential(
            CommonBlock(64, 64, 1),
            CommonBlock(64, 64, 1),
            CommonBlock(64, 64, 1)
        )
        self.layer2 = nn.Sequential(
            SpecialBlock(64, 128, [2, 1]),
            CommonBlock(128, 128, 1),
            CommonBlock(128, 128, 1),
            CommonBlock(128, 128, 1)
        )
        self.layer3 = nn.Sequential(
            SpecialBlock(128, 256, [2, 1]),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1)
        )
        self.layer4 = nn.Sequential(
            SpecialBlock(256, 512, [2, 1]),
            CommonBlock(512, 512, 1),
            CommonBlock(512, 512, 1)
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(960, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, classes_num)
        )
        self.con1 = nn.Conv2d(384, 128, 3, 1, 1)
        self.bn = nn.BatchNorm2d(128)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cbam1 = CBAMLayer(channel=64)
        self.cbam2 = CBAMLayer(channel=64)
        self.cbam3 = CBAMLayer(channel=128)
        self.cbam4 = CBAMLayer(channel=256)
        self.att = EfficientChannelAttention(960)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(960, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes_num)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x_rgb = self.prepare(x)  # 三通道 64
        x = self.layer1(x_rgb)
        x = self.cbam2(x) * x + x  # 16
        x1 = self.avg_pool(x)

        x = self.layer2(x)
        x = self.cbam3(x) * x + x  # 8
        x2 = self.avg_pool(x)

        x = self.layer3(x)
        x = self.cbam4(x) * x + x  # 4
        x3 = self.avg_pool(x)

        x = self.layer4(x)  # 2
        x = self.avg_pool(x)

        x = torch.cat((x, x1, x2, x3), dim=1)
        x = x * self.att(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        return x


if __name__ == "__main__":
    model = ResNet34(classes_num=2).to(device="cuda")
    x = torch.zeros((1, 3, 112, 112), device="cuda")
    # summary(model, (x, x, x))
    output = model(x)
    print(output)
