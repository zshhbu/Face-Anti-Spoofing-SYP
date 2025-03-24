# 数据增强和测试指标的代码集中在这里
# 导入必备的包
import math
import random

import numpy as np
# 网络模型构建需要的包
import torch
import torchvision
# import timm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from collections import defaultdict

# Metric 测试准确率需要的包
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_curve
# Augmentation 数据增强要使用到的包
# import albumentations
# from albumentations.pytorch.transforms import ToTensorV2conda
from torchvision import datasets, models, transforms


# 这个库主要用于定义如何进行数据增强。
# https://zhuanlan.zhihu.com/p/149649900?from_voters_page=true


# 训练集的预处理以及数据增强


# 验证集和测试集的预处理
# def get_valid_transforms(img_size=224):
#     return albumentations.Compose(
#         [
#             albumentations.Resize(img_size, img_size),
#             albumentations.Normalize(
#                 [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
#                 max_pixel_value=255.0, always_apply=True
#             ),
#             ToTensorV2(p=1.0)
#         ]
#     )


class MergeItems(object):
    def __init__(self, same_label=False, p=0.5):
        self.p = p
        self.same_label = same_label

    def __call__(self, img_dict):

        if np.random.rand() < self.p:
            data_get_func = img_dict['meta']['get_item_func']
            curr_idx = img_dict['meta']['idx']
            max_idx = img_dict['meta']['max_idx']

            other_idx = np.random.randint(0, max_idx)
            data4augm = data_get_func(other_idx)
            while (curr_idx == other_idx) or (self.same_label and data4augm['label'] != img_dict['label']):
                other_idx = np.random.randint(0, max_idx)
                data4augm = data_get_func(other_idx)

            alpha = np.random.rand()

            keys = ['rgb', 'depth', 'ir']
            for key in keys:
                img_dict[key] = Image.blend(data4augm[key].resize(img_dict[key].size),
                                            img_dict[key],
                                            alpha=alpha)
            if not self.same_label:
                img_dict['label'] = alpha * img_dict['label'] + (1 - alpha) * data4augm['label']

        return img_dict

    def __repr__(self):
        return self.__class__.__name__ + '(same_label={0}, p={1})'.format(self.same_label, self.p)


def get_torch_transforms(img_size=224):
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(img_size),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomRotation((-5, 5)),
            transforms.RandomAutocontrast(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # transforms.Resize((img_size, img_size)),
            # transforms.Resize(256),
            # transforms.CenterCrop(img_size),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms


import os
from PIL import Image
from torch.utils.data import Dataset


class CISIA_surf(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform
        self.data = self._load_data(file_path)
        # self.parent_dir = os.path.dirname(file_path)

    def __getitem__(self, index):
        rgb_path, dp_path, ir_path, label = self.data[index]

        # 读取图像
        rgb_image = Image.open(rgb_path).convert("RGB")
        dp_image = Image.open(dp_path).convert("L").convert("RGB")  # 转为灰度图像
        # print(len(dp_image.split()))# 查看通道数

        ir_image = Image.open(ir_path).convert("L").convert("RGB")  # 转换为灰度图像

        # 进行数据增强
        if self.transform:
            rgb_image = self.transform(rgb_image)
            dp_image = self.transform(dp_image)
            ir_image = self.transform(ir_image)
        return rgb_image, dp_image, ir_image, label

    def __len__(self):
        return len(self.data)

    def _load_data(self, file_path):
        self.parent_dir = os.path.dirname(file_path)
        data = []
        with open(self.file_path, 'r') as file:
            for line in file.readlines():
                rgb_path, dp_path, ir_path, label = line.split()
                rgb_path = os.path.join(self.parent_dir, line.split()[0]).replace('/', '\\')
                dp_path = os.path.join(self.parent_dir, line.split()[1]).replace('/', '\\')
                ir_path = os.path.join(self.parent_dir, line.split()[2]).replace('/', '\\')
                data.append((rgb_path, dp_path, ir_path, label))
        return data


class CISIA_surf1(Dataset):
    def __init__(self, file_path, transform=None, train_ratio=0.7, is_training=True):
        self.file_path = file_path
        self.transform = transform
        self.data = self._load_data(file_path)
        self.train_ratio = train_ratio
        self.is_training = is_training
        # self.parent_dir = os.path.dirname(file_path)

    def __getitem__(self, index):
        rgb_path, dp_path, ir_path, label = self.data[index]

        # 读取图像
        rgb_image = Image.open(rgb_path).convert("RGB")
        dp_image = Image.open(dp_path).convert("L").convert("RGB")  # 转为灰度图像
        # print(len(dp_image.split()))# 查看通道数

        ir_image = Image.open(ir_path).convert("L").convert("RGB")  # 转换为灰度图像

        # 进行数据增强
        if self.transform:
            rgb_image = self.transform(rgb_image)
            dp_image = self.transform(dp_image)
            ir_image = self.transform(ir_image)
        return rgb_image, dp_image, ir_image, label

    def __len__(self):
        return len(self.data)

    def _load_data(self, file_path, train_ratio=0.7, is_training=True):
        self.parent_dir = os.path.dirname(file_path)
        self.train_ratio = train_ratio
        self.is_training = is_training
        data = []
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
            # 计算划分训练集和测试集的索引位置
            total_lines = len(lines)
            train_size = int(total_lines * self.train_ratio)
            # 使用 random.shuffle() 函数对列表进行随机打乱
            random.shuffle(lines)
            # 划分训练集和测试集
            if self.is_training:
                selected_lines = lines[:train_size]
            else:
                selected_lines = lines[train_size:]
            for line in selected_lines:
                rgb_path, dp_path, ir_path, label = line.split()
                rgb_path = os.path.join(self.parent_dir, line.split()[0]).replace('/', '\\')
                dp_path = os.path.join(self.parent_dir, line.split()[1]).replace('/', '\\')
                ir_path = os.path.join(self.parent_dir, line.split()[2]).replace('/', '\\')
                data.append((rgb_path, dp_path, ir_path, label))
        return data


class CISIA_cefA(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform
        self.data = self._load_data(file_path)
        # self.parent_dir = os.path.dirname(file_path)

    def __getitem__(self, index):
        rgb_path, dp_path, ir_path, label = self.data[index]

        # 读取图像
        rgb_image = Image.open(rgb_path).convert("RGB")
        dp_image = Image.open(dp_path).convert("L").convert("RGB")  # 转为灰度图像
        # print(len(dp_image.split()))# 查看通道数

        ir_image = Image.open(ir_path).convert("L").convert("RGB")  # 转换为灰度图像

        # 进行数据增强
        if self.transform:
            rgb_image = self.transform(rgb_image)
            dp_image = self.transform(dp_image)
            ir_image = self.transform(ir_image)
        return rgb_image, dp_image, ir_image, label

    def __len__(self):
        return len(self.data)

    def _load_data(self, file_path):
        self.parent_dir = os.path.dirname(file_path)
        data = []
        with open(self.file_path, 'r') as file:
            for line in file.readlines():
                rgb_path, label = line.split()
                dpPath = rgb_path.replace('profile', 'depth')
                irPath = rgb_path.replace('profile', 'ir')
                rgb_path = os.path.join(self.parent_dir, rgb_path).replace('/', '\\')
                dp_path = os.path.join(self.parent_dir, dpPath).replace('/', '\\')
                ir_path = os.path.join(self.parent_dir, ir_path).replace('/', '\\')
                data.append((rgb_path, dp_path, ir_path, label))
        return data


# # 定义数据增强的转换操作
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # 调整图像大小为指定尺寸
#     transforms.RandomHorizontalFlip(),  # 随机水平翻转
#     transforms.ToTensor(),  # 将图像转换为张量
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化图像
# ])
#
# # 数据集文件路径
# file_path = r"D:\BaiduNetdiskDownload\FAS\challenge\phase1\train_list.txt"
# # 创建自定义数据集实例
# dataset = CISIA_surf(file_path, transform=transform)
#
# # 获取数据样本
# sample = dataset[0]
# rgb_data, dp_data, ir_data, label = sample
#
# # 打印数据
# print("RGB 数据:", rgb_data)
# print("深度数据:", dp_data)
# print("红外数据:", ir_data)
# print("标签:", label)

import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class Casia_fasd(Dataset):
    def __init__(self, path, train_ratio=0.7, transform=None, is_training=True, is_testing=False):
        self.path = path
        self.ratio = train_ratio
        self.transform = transform
        self.training = is_training
        self.testing = is_testing
        self.image_files = [file for file in os.listdir(self.path)]
        random.shuffle(self.image_files)
        self.total_images = len(self.image_files)

        # Split the data into training and validation sets
        train_size = int(self.total_images * self.ratio)
        self.train_images = self.image_files[:train_size]
        self.val_images = self.image_files[train_size:]
        self.test_images = self.image_files

    def __getitem__(self, index):
        img_list = self.train_images if self.training else self.val_images
        img_list = self.test_images if self.testing else img_list

        # Get the image file name and label
        img_name = img_list[index]
        label = int(img_name[-8:-4] == "real")
        # Load the image using PIL
        image = Image.open(os.path.join(self.path, img_name))

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        if self.training:
            return len(self.train_images)
        elif not self.training and not self.testing:
            return len(self.val_images)
        else:
            return len(self.test_images)


class Casia_fasd1(Dataset):
    def __init__(self, rgb_path, lbp_path, train_ratio=0.7, transform=None, is_training=True, is_testing=False):
        self.path = rgb_path
        self.path1 = lbp_path
        self.ratio = train_ratio
        self.transform = transform
        self.training = is_training
        self.testing = is_testing
        self.image_files = [file for file in os.listdir(self.path)]
        self.image_files1 = [file for file in os.listdir(self.path1)]
        self.total_images = len(self.image_files)

        # Split the data into training and validation sets
        train_size = int(self.total_images * self.ratio)
        self.train_images = self.image_files[:train_size]
        self.val_images = self.image_files[train_size:]
        self.test_images = self.image_files
        self.train_images1 = self.image_files1[:train_size]
        self.val_images1 = self.image_files1[train_size:]
        self.test_images1 = self.image_files1

    def __getitem__(self, index):
        img_list = self.train_images if self.training else self.val_images
        img_list = self.test_images if self.testing else img_list
        img_list1 = self.train_images1 if self.training else self.val_images1
        img_list1 = self.test_images1 if self.testing else img_list1
        # Get the image file name and label
        img_name = img_list[index]
        img_name1 = img_list1[index]
        label = int(img_name[-8:-4] == "real")
        # Load the image using PIL
        image = Image.open(os.path.join(self.path, img_name))
        image1 = Image.open(os.path.join(self.path1, img_name1))
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
            image_lbp = self.transform(image1)
        return image, image_lbp, label

    def __len__(self):
        if self.training:
            return len(self.train_images)
        elif not self.training and not self.testing:
            return len(self.val_images)
        else:
            return len(self.test_images)


class Casia_fasd2(Dataset):
    def __init__(self, rgb_path, lbp_path, train_ratio=0.7, transform=None, graytransform=None, is_training=True,
                 is_testing=False):
        self.path = rgb_path
        self.path1 = lbp_path
        self.ratio = train_ratio
        self.transform = transform
        self.graytransform = graytransform
        self.training = is_training
        self.testing = is_testing
        self.image_files = [file for file in os.listdir(self.path)]
        self.image_files1 = [file for file in os.listdir(self.path1)]
        self.total_images = len(self.image_files)

        # Split the data into training and validation sets
        train_size = int(self.total_images * self.ratio)
        self.train_images = self.image_files[:train_size]
        self.val_images = self.image_files[train_size:]
        self.test_images = self.image_files
        self.train_images1 = self.image_files1[:train_size]
        self.val_images1 = self.image_files1[train_size:]
        self.test_images1 = self.image_files1

    def __getitem__(self, index):
        img_list = self.train_images if self.training else self.val_images
        img_list = self.test_images if self.testing else img_list
        img_list1 = self.train_images1 if self.training else self.val_images1
        img_list1 = self.test_images1 if self.testing else img_list1
        # Get the image file name and label
        img_name = img_list[index]
        img_name1 = img_list1[index]
        label = int(img_name[-8:-4] == "real")
        # Load the image using PIL
        image = Image.open(os.path.join(self.path, img_name))
        image1 = Image.open(os.path.join(self.path1, img_name1))
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        if self.graytransform:
            image_lbp = self.graytransform(image1)
        return image, image_lbp, label

    def __len__(self):
        if self.training:
            return len(self.train_images)
        elif not self.training and not self.testing:
            return len(self.val_images)
        else:
            return len(self.test_images)


class MSU_MFSD(Dataset):
    def __init__(self, Path, is_train=True, is_test=False, transform=None):
        self.path = Path
        self.training = is_train
        self.testing = is_test
        self.transform = transform

        with open(self.path, 'r') as file:
            content = file.read()
        # 将内容拆分成行，每行表示一个数据项
        shuffled_files = content.splitlines()
        # 对数据项进行随机打乱
        random.shuffle(shuffled_files)
        train_ratio = 0.6
        test_ratio = 0.3
        # 计算切分点的索引位置
        train_cutoff = int(train_ratio * len(shuffled_files))
        test_cutoff = train_cutoff + int(test_ratio * len(shuffled_files))

        # 划分成训练集、测试集和验证集
        self.train_files = shuffled_files[:train_cutoff]
        self.val_files = shuffled_files[train_cutoff:test_cutoff]
        self.test_files = shuffled_files[test_cutoff:]
        # print('train:', len(self.train_files), "val:", len(self.val_files), 'test:', len(self.test_files))

    def __getitem__(self, item):
        if self.training:
            image_path = os.path.join(self.path, self.train_files[item])
        elif not self.training and not self.testing:
            image_path = os.path.join(self.path, self.val_files[item])
        else:
            image_path = os.path.join(self.path, self.test_files[item])
        image = Image.open(image_path)
        label = '1' if image_path[46:50] == "real" else '0'
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        if self.training:
            return len(self.train_files)
        elif not self.training and not self.testing:
            return len(self.val_files)
        elif self.testing:
            return len(self.test_files)


class ReplayAttack(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.file_paths = []  # 存储文件路径的列表
        self.labels = []  # 存储标签的列表

        with open(self.path, 'r') as file:
            for line in file.readlines():
                file_path = line.split('，')[0]
                self.file_paths.append(file_path)
                label = line.split('，')[-1]
                self.labels.append(label)
                # file_path = line[0:-2]
                # self.file_paths.append(file_path)
                # label = line[-2]
                # self.labels.append(label)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx])  # 根据索引打开对应的图像文件
        label = self.labels[idx]  # 根据索引获取对应的标签

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.file_paths)  # 返回数据集的总样本数

import os
from PIL import Image
from torch.utils.data import Dataset


class OULU_NPU(Dataset):
    """双输入"""
    def __init__(self, path, h_path, transform, gray_transform, protocol, testing):
        self.path = path
        self.h_path = h_path
        self.transform = transform
        self.gray_transform = gray_transform
        self.protocol = protocol
        self.testing = testing
        self.file_paths = []
        self.labels = []
        self.file_paths1 = []

        self.load_file_paths_and_labels()

    def load_file_paths_and_labels(self):
        for dirpath, _, filenames in os.walk(self.path):
            for filename in filenames:
                label = '1' if dirpath[-1] == '1' else '0'  # Assuming label is the first part of the filename
                file_path = os.path.join(dirpath, filename)

                if self.protocol == '1':
                    if not self.testing and dirpath[-6] in ['1', '2']:
                        self.file_paths.append(file_path)
                        self.labels.append(label)
                    elif self.testing and dirpath[-6] == '3':
                        self.file_paths.append(file_path)
                        self.labels.append(label)
                elif self.protocol == '2':
                    if not self.testing and dirpath[-1] in ['2', '4']:
                        self.file_paths.append(file_path)
                        self.labels.append(label)
                    elif self.testing and dirpath[-1] in ['3', '5']:
                        self.file_paths.append(file_path)
                        self.labels.append(label)
                elif self.protocol == '3':
                    if not self.testing and dirpath[-8] != '6':
                        self.file_paths.append(file_path)
                        self.labels.append(label)
                    elif self.testing and dirpath[-8] == '6':
                        self.file_paths.append(file_path)
                        self.labels.append(label)
                elif self.protocol == '4':
                    if not self.testing and dirpath[-1] in ['2', '4'] and dirpath[-6] in ['1', '2'] and dirpath[-8] != '6':
                        self.file_paths.append(file_path)
                        self.labels.append(label)
                    elif self.testing and dirpath[-1] in ['3', '5'] and dirpath[-6] == '3' and dirpath[-8] == '6':
                        self.file_paths.append(file_path)
                        self.labels.append(label)
        for dirpath, _, filenames in os.walk(self.h_path):
            for filename in filenames:
                label = '1' if dirpath[-1] == '1' else '0'  # Assuming label is the first part of the filename
                file_path = os.path.join(dirpath, filename)

                if self.protocol == '1':
                    if not self.testing and dirpath[-6] in ['1', '2']:
                        self.file_paths1.append(file_path)
                        self.labels.append(label)
                    elif self.testing and dirpath[-6] == '3':
                        self.file_paths1.append(file_path)
                        self.labels.append(label)
                elif self.protocol == '2':
                    if not self.testing and dirpath[-1] in ['2', '4']:
                        self.file_paths1.append(file_path)
                        self.labels.append(label)
                    elif self.testing and dirpath[-1] in ['3', '5']:
                        self.file_paths1.append(file_path)
                        self.labels.append(label)
                elif self.protocol == '3':
                    if not self.testing and dirpath[-8] != '6':
                        self.file_paths1.append(file_path)
                        self.labels.append(label)
                    elif self.testing and dirpath[-8] == '6':
                        self.file_paths1.append(file_path)
                        self.labels.append(label)
                elif self.protocol == '4':
                    if not self.testing and dirpath[-1] in ['2', '4'] and dirpath[-6] in ['1', '2'] and dirpath[-8] != '1':
                        self.file_paths1.append(file_path)
                        self.labels.append(label)
                    elif self.testing and dirpath[-1] in ['3', '5'] and dirpath[-6] == '3' and dirpath[-8] == '1':
                        self.file_paths1.append(file_path)
                        self.labels.append(label)

    def __getitem__(self, index):
        img = Image.open(self.file_paths[index])
        h_img = Image.open(self.file_paths1[index])
        label = self.labels[index]

        if self.transform:
            img = self.transform(img)
        if self.gray_transform:
            h_img = self.gray_transform(h_img)

        return img, h_img, label

    def __len__(self):
        return len(self.file_paths)


class OULU_NPU1(Dataset):
    """单输入"""
    def __init__(self, path, transform, protocol, testing):
        self.path = path
        self.transform = transform
        self.protocol = protocol
        self.testing = testing
        self.file_paths = []
        self.labels = []
        self.load_file_paths_and_labels()
        selected_file_paths = []
        selected_labels = []

    def load_file_paths_and_labels(self):
        # for dirpath, _, filenames in os.walk(self.path):
        #     for filename in filenames:
        #         label = '1' if dirpath[-1] == '1' else '0'  # Assuming label is the first part of the filename
        #         file_path = os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(self.path):
            for index, filename in enumerate(filenames):
                label = '1' if dirpath[-1] == '1' else '0'  # 假设标签是文件名的第一部分
                file_path = os.path.join(dirpath, filename)

                if self.protocol == '1':
                    if not self.testing and dirpath[-6] in ['1', '2']:
                        # if dirpath[-1] != '1' and index % 3 == 0:  # 每隔三张图片读入一张
                        #     if dirpath[-1] == '1':
                        #         self.file_paths.append(file_path)
                        #         self.labels.append(label)
                        #     elif dirpath[-1] != '1' and index % 5 == 0:   # 每隔三张图片读入一张
                        self.file_paths.append(file_path)
                        self.labels.append(label)
                    elif self.testing and dirpath[-6] == '3':
                        # if dirpath[-1] == '1' and index % 20 == 0:
                        #     self.file_paths.append(file_path)
                        #     self.labels.append(label)
                        # elif dirpath[-1] != '1' and index % 10 == 0:   # 每隔三张图片读入一张
                        #     self.file_paths.append(file_path)
                        #     self.labels.append(label)
                        self.file_paths.append(file_path)
                        self.labels.append(label)

                elif self.protocol == '2':
                    if not self.testing and dirpath[-1] in ['1', '2', '4']:
                        self.file_paths.append(file_path)
                        self.labels.append(label)
                    elif self.testing and dirpath[-1] in ['1', '3', '5']:
                        self.file_paths.append(file_path)
                        self.labels.append(label)
                elif self.protocol == '3':
                    if not self.testing and dirpath[-8] != '1':
                        self.file_paths.append(file_path)
                        self.labels.append(label)
                    elif self.testing and dirpath[-8] == '1':
                        self.file_paths.append(file_path)
                        self.labels.append(label)
                elif self.protocol == '4':
                    if not self.testing and dirpath[-1] in ['1', '2', '4'] and dirpath[-6] in ['1', '2'] and dirpath[-8] != '1':
                        self.file_paths.append(file_path)
                        self.labels.append(label)
                    elif self.testing and dirpath[-1] in ['1', '3', '5'] and dirpath[-6] == '3' and dirpath[-8] == '1':
                        self.file_paths.append(file_path)
                        self.labels.append(label)

    def __getitem__(self, index):
        img = Image.open(self.file_paths[index])
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.file_paths)


if __name__ == '__main__':
    # path = r"D:\BaiduNetdiskDownload\cssia_fasd\test_img\test_img\color"
    # test_dataset = Casia_fasd(path, is_training=False, train_ratio=0.7, is_testing=True)
    # batch_size = 32
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # print(len(test_dataset))

    # path = r"D:\BaiduNetdiskDownload\MSU-MFSD\scene01\path.txt"
    # transform = transforms.Compose([transforms.Resize((64, 64)),  # cannot 224, must (224, 224)
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    # test = MSU_MFSD(path, is_train=False, is_test=False, transform=transform)
    # print(len(test))

    # data_loader = DataLoader(test, batch_size=1, shuffle=True)
    # for images, labels in data_loader:
    #     print(labels)

    path = r"D:\BaiduNetdiskDownload\oulu_npu\train"
    transform = transforms.Compose([transforms.Resize((64, 64)),  # cannot 224, must (224, 224)
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    test = OULU_NPU(path, transform=transform, testing=False, protocol='1')
    dataloader = DataLoader(test, batch_size=1, shuffle=False)
    for image, label in dataloader:
        print(label)


# 测试准确率
def accuracy(output, target):
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    target = target.cpu()

    return accuracy_score(target, y_pred)


# 计算f1
def calculate_f1_macro(output, target):
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    target = target.cpu()

    return f1_score(target, y_pred, average='macro')


# 计算recall
def calculate_recall_macro(output, target):
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    target = target.cpu()
    # tp fn fp
    return recall_score(target, y_pred, average="macro", zero_division=0)


# 训练的时候输出信息使用
class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"],
                    float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


# 调整学习率
def adjust_learning_rate(optimizer, epoch, params, batch=0, nBatch=None):
    """ adjust learning of a given optimizer and return the new learning rate """
    new_lr = calc_learning_rate(epoch, params['lr'], params['epoch'], batch, nBatch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


""" learning rate schedule """


# 计算学习率
def calc_learning_rate(epoch, init_lr, n_epochs, batch=0, nBatch=None, lr_schedule_type='cosine'):
    if lr_schedule_type == 'cosine':
        t_total = n_epochs * nBatch
        t_cur = epoch * nBatch + batch
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * t_cur / t_total))
    elif lr_schedule_type is None:
        lr = init_lr
    else:
        raise ValueError('do not support: %s' % lr_schedule_type)
    return lr


import numpy as np


def error_rate(predictions, labels):
    """
    计算错误率

    参数:
        predictions (list or numpy array): 模型的预测结果列表或数组
        labels (list or numpy array): 真实的标签列表或数组

    返回:
        float: 错误率，范围在[0, 1]之间
    """
    if len(predictions) != len(labels):
        raise ValueError("预测结果和标签数量不一致")

    total_samples = len(predictions)
    incorrect_samples = sum(1 for pred, label in zip(predictions, labels) if pred != label)
    error_rate = incorrect_samples / total_samples

    return error_rate


def calculate_eer(scores, labels):
    """
    计算等错误率（Equal Error Rate，EER）

    参数:
        scores (list or numpy array): 模型输出的概率或得分列表或数组
        labels (list or numpy array): 真实的标签列表或数组（0或1）

    返回:
        float: 等错误率
    """

    if len(scores) != len(labels):
        raise ValueError("模型输出的概率或得分和标签数量不一致")
    scores = torch.softmax(scores, dim=1)
    scores = torch.argmax(scores, dim=1).cpu()
    labels = labels.cpu()
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    # 寻找FPR和FNR最接近的阈值
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    # 在等错误率的阈值下计算错误率
    predicted_labels = np.where(scores >= eer_threshold, 1, 0)
    eer = error_rate(predicted_labels, labels)

    return eer


"""
 计算 HTER，平均 FAR 和 FRR
"""


def calculate_far_frr(true_labels, predicted_labels):
    total_samples = len(true_labels)
    false_acceptances = 0  # 记录假阳性的数量
    false_rejections = 0  # 记录假阴性的数量

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        # 假阳性：真实为负样本，但被错误地分类为正样本
        if true_label == 0 and predicted_label == 1:
            false_acceptances += 1
        # 假阴性：真实为正样本，但被错误地分类为负样本
        elif true_label == 1 and predicted_label == 0:
            false_rejections += 1

    # 计算 FAR 和 FRR
    far = false_acceptances / total_samples
    frr = false_rejections / total_samples

    return far, frr


def calculate_hter(far, frr):
    hter = (far + frr) / 2
    return hter


"""
计算APCER BPCER ACER 平均分类错误率
"""


def calculate_apcer(predictions, labels, threshold):
    # 计算在攻击样本中错误地被分类为真实人脸的比例 (APCER)
    # num_attack_samples = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred < threshold)
    num_attack_samples = 0
    for index, i in enumerate(predictions):
        if (labels[index] == 0) & (i[0] < threshold):
            num_attack_samples += 1
    #
    # for pred in predictions:
    #     if pred[0] < threshold and pred[1] < threshold or pred[0] > threshold and pred[1] > threshold:
    #         # 如果两个数都小于阈值，返回较大数的索引
    #         if pred[0] > pred[1]:
    #             predicted_classes.append(0)  # 返回第一个数的索引
    #         else:
    #             predicted_classes.append(1)  # 返回第二个数的索引
    #     else:
    #         # 如果两个值都小于阈值，比较它们的大小
    #         if pred[0] > threshold:
    #             predicted_classes.append(0)  # 如果第一个值大于第二个值，返回1
    #         else:
    #             predicted_classes.append(1)  # 如果第二个值大于第一个值，返回0
    #
    # num_attack_samples = np.sum(np.logical_and(labels == 0, predicted_classes == 1))
    num_all_attack_samples = sum(1 for label in labels if label == 1)
    apcer = num_attack_samples / num_all_attack_samples
    return apcer


def calculate_npcer(predictions, labels, threshold):
    # predicted_classes = []
    #
    # for pred in predictions:
    #     if pred[0] < threshold and pred[1] < threshold or pred[0] > threshold and pred[1] > threshold:
    #         # 如果两个数都小于阈值，返回较大数的索引
    #         if pred[0] > pred[1]:
    #             predicted_classes.append(0)  # 返回第一个数的索引
    #         else:
    #             predicted_classes.append(1)  # 返回第二个数的索引
    #     else:
    #         # 如果两个值都小于阈值，比较它们的大小
    #         if pred[0] > threshold:
    #             predicted_classes.append(0)  # 如果第一个值大于第二个值，返回1
    #         else:
    #             predicted_classes.append(1)  # 如果第二个值大于第一个值，返回0
    # num_wrongly_classified_as_attack = np.sum(np.logical_and(labels == 1, predicted_classes == 0))
    # 计算在真实人脸样本中错误地被分类为攻击的比例 (NPCER)
    num_genuine_samples = sum(1 for label in labels if label == 0)
    num_wrongly_classified_as_attack = 0
    for index, i in enumerate(predictions):
        if (labels[index] == 1) & (i[0] >= threshold):
            num_wrongly_classified_as_attack += 1
    npcer = num_wrongly_classified_as_attack / num_genuine_samples
    return npcer
def calculate_acer(apcer, npcer):
    # 计算平均分类错误率 (ACER)
    acer = (apcer + npcer) / 2
    return acer

def test_threshold_based(predictions, labels, threshold):


    # type1 = 0.0
    # type2 = 0.0
    # for index, i in enumerate(predictions):
    #     if (labels[index] == 1) & (i[0] > threshold):
    #         type2 += 1
    #     if (labels[index] == 0) & (i[1] > (1-threshold)):
    #         type1 += 1
    # num_real = sum(1 for label in labels if label == 1)
    # num_fake = sum(1 for label in labels if label == 0)
    # ACC = 1 - (type1 + type2) / len(labels)
    # APCER = type2 / num_real
    # BPCER = type1 / num_fake
    # ACER = (APCER + BPCER) / 2.0
    type1 = 0.0
    type2 = 0.0
    # 计算在攻击样本中错误地被分类为真实人脸的比例 (APCER)
    for index, i in enumerate(predictions):
        if (labels[index] == 0) & (i[1] > threshold):
            type1 += 1
    # 计算在真实人脸样本中错误地被分类为攻击的比例 (NPCER)
    for index, i in enumerate(predictions):
        if (labels[index] == 1) & (i[0] >= threshold):
            type2 += 1
    num_real = sum(1 for label in labels if label == 1)
    num_fake = sum(1 for label in labels if label == 0)
    far = type1/num_fake
    frr = type2/num_real
    hter = (far + frr) / 2
    ACC = 1 - (type1 + type2) / len(labels)
    APCER = type1 / num_fake
    BPCER = type2 / num_real
    ACER = (APCER + BPCER) / 2.0
    return ACC, APCER, BPCER, ACER, hter



