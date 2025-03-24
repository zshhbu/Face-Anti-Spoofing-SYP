import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
# from Model.resnet_1120 import ResNet18
from Model.resnet18_chidu import ResNet18
# from Model.resnet18 import ResNet18
from cam_utils import *
import cv2
import numpy as np


def main():
    model = ResNet18()  # 导入自己的模型，num_classes数为自己数据集的类别数
    weights_dict = torch.load(r"D:\BaiduNetdiskDownload\CDCN-master\CDCN-master\checkpoints\RA-chidu-12-18\Best.pth", map_location='cpu')

    model.load_state_dict(weights_dict, strict=False)  # 加载自己模型的权重
    # target_layers = [model.backbone.layer4]
    target_layers = [model.layer4[-1]]  # 定义目标层


    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    img_path = r"D:\BaiduNetdiskDownload\replayattack\test\attack_cut2\fixeattack_highdef_client009_session01_highdef_photo_adverse\frame_0001.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img = cv2.resize(img, (224, 224))  # 将导入的图片reshape到自己想要的尺寸

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    # target_category = 281  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    label = [1, 0]
    target_category = torch.LongTensor(label)
    grayscale_cam = cam(input_tensor=input_tensor, target=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()