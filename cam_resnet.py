from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
import cv2
from cam_utils import *
# from Model.resnet_1120 import ResNet18
# from Model.resnet18 import ResNet18
from  Model.resnet18_att import ResNet18
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Full_layer(torch.nn.Module):
    '''explicitly define the full connected layer'''

    def __init__(self, feature_num, class_num):
        super(Full_layer, self).__init__()
        self.class_num = class_num
        self.fc = nn.Linear(feature_num, class_num)

    def forward(self, x):
        x = self.fc(x)
        return x


fc = Full_layer(1024, 3)


class Cmodule(nn.Module):
    def __init__(self, submodel1):
        super(Cmodule, self).__init__()
        self.submodel1 = submodel1
        # self.submodel2 = submodel2

    def forward(self, x):
        x = self.submodel1(x)
        # x = self.submodel2(x)
        return x


# 1）建立模型、加载预训练参数
submodel1 = ResNet18()
# submodel2 = fc
# model = Cmodule(submodel1, submodel2)
model = Cmodule(submodel1)
# 加载预训练模型的状态字典
state_dict = torch.load("D:\BaiduNetdiskDownload\CDCN-master\CDCN-master\checkpoints\RA-att-12-15\Best.pth")
# 去掉参数名中的"module."前缀
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
# 在你的模型实例上加载状态字典
model.submodel1.load_state_dict(state_dict)

img_g_rgb_path = r"D:\BaiduNetdiskDownload\replayattack\test\attack_cut2\fixeattack_highdef_client009_session01_highdef_photo_adverse\frame_0002.png"# CASME_onset_apex_new/sub02/EP13_04/reg_img51.jpg
img_rgb = cv2.imread(img_g_rgb_path)
img_rgb = cv2.resize(img_rgb, (64, 64))
img_rgb = torch.Tensor(np.array(img_rgb)).permute(2, 0, 1)
data_transform = transforms.Compose([
                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
img_rgb = data_transform(img_rgb)

src_tensor = torch.unsqueeze(img_rgb, dim=0)
print(model(src_tensor))
# 这里是因为模型接受的数据维度是[B,C,H,W]，输入的只有一张图片所以需要升维
label = [1, 0]
# label_numpy = np.array(label)
# gt_tensor = torch.from_numpy(label_numpy)
gt_tensor = torch.LongTensor(label)
print(gt_tensor)

# 3）指定需要计算CAM的网络结构
target_layers = [model.submodel1.layer4]  # down4()是在Net网络中__init__()方法中定义了的self.down4

cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
grayscale_cam = cam(input_tensor=src_tensor, target=gt_tensor)

grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(img_rgb / 255., grayscale_cam, use_rgb=True)

plt.imshow(visualization)
plt.show()
plt.close('all')
