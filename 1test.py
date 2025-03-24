#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：cls_torch_tem 
@File    ：val.py
@Author  ：ChenmingSong
@Date    ：2022/3/29 10:07 
@Description：验证模型的准确率
'''
import time
import warnings

warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt

# 最好是把配置文件写在一起，如果写在一起的话，方便进行查看
from torchutils import *
from torchvision import transforms
import os
# from Model.resnet_1023 import ResNet18  # 两个模块都有
# from Model.resnet18 import ResNet18   #原始
# from Model.resnet18_att import ResNet18
# from Model.resnet18_chidu import ResNet18
# from Model.resnet_1120 import ResNet18
# from Model.r1esnet18 import ResNet18
# from Model.resnet_maxpool import ResNet18
from Model.resnet_chiduxiaorong import ResNet18
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')
# 固定随机种子，保证实验结果是可以复现的
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# data_path = r'D:\BaiduNetdiskDownload\cssia_fasd\train_img\train_img\color'  # todo 修改为数据集根目录  CAISA_FASD
# data_path = r"D:\BaiduNetdiskDownload\MSU-MFSD\scene01\path.txt"             # todo  MSU_MFSD
data_path = r"D:\BaiduNetdiskDownload\replayattack\test\test.txt"              # todo ReplayAttack
# data_path = r"D:\BaiduNetdiskDownload\oulu_npu\test"
model_path = r"checkpoints/RA-x1 2 4-4-10/Best.pth"  # todo 模型地址
model_name = 'resnet18'  # todo 模型名称
img_size = 64 # todo 数据集训练时输入模型的大小
# 注： 执行之前请先划分数据集
# 超参数设置
params = {
    'model': model_name,  # 选择预训练模型
    "img_size": img_size,  # 图片输入大小
    "test_dir": data_path,  # todo 测试集子目录
    'device': device,  # 设备
    'batch_size': 128,  # 批次大小
    'num_workers': 0,  # 进程
    "num_classes": 2,  # 类别数目, 自适应获取类别数目
}


def test(val_loader, model, params):
    metric_monitor = MetricMonitor()  # 验证流程
    model.eval()  # 模型设置为验证格式
    stream = tqdm(val_loader)  # 设置进度条

    # 对模型分开进行推理
    test_real_labels = []
    test_pre_labels = []

    with torch.no_grad():  # 开始推理
        for i, images in enumerate(stream, start=1):
            rgb_data, label = images
            rgb_image = rgb_data.to(params['device'], non_blocking=True)
            label = [int(l) for l in label]
            target = torch.tensor(label).to(params['device'], non_blocking=True)
            output = model(rgb_image)  ################# 前向传播
            target_numpy = target.cpu().numpy()
            y_pred = torch.softmax(output, dim=1)
            predict = y_pred.cpu().numpy()
            y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()

            # for depth in depth_all:
            #     dep = depth.cpu().numpy()
            #     dep = np.squeeze(dep)
            #     plt.imshow(dep, cmap='viridis', vmin=0, vmax=1)  # 使用'viridis'颜色映射，根据需要调整最小和最大值
            #     plt.colorbar()  # 添加颜色条
            #     plt.show()
            test_real_labels.extend(target_numpy)
            test_pre_labels.extend(y_pred)
            f1_macro = calculate_f1_macro(output, target)  # 计算f1分数
            eer = calculate_eer(output, target)  # 计算eer``
            far, frr = calculate_far_frr(target, y_pred)  # 计算 FAR 和 FRR
            hter = calculate_hter(far, frr)  # 计算 HTER
            apcer = calculate_apcer(predict, target_numpy, threshold=0.3)  # 计算APCER   predict
            bpcer = calculate_npcer(predict, target_numpy, threshold=0.7)  # 计算BPCER
            acer = calculate_acer(apcer, bpcer)  # 计算ACER
            acc = accuracy(output, target)  # 计算acc
            # acc, apcer, bpcer, acer, hter = test_threshold_based(predict, target_numpy, 0.5)   ## 0.726 1; # 0.726 2; #0.731 3-1;
            recall_macro = calculate_recall_macro(output, target)  # 计算recall分数

            # metric_monitor.update('Loss', loss.item())  # 后面基本都是更新进度条的操作
            metric_monitor.update('F1', f1_macro)
            metric_monitor.update("Recall", recall_macro)
            metric_monitor.update('Accuracy', acc)
            metric_monitor.update('eer', eer)
            metric_monitor.update('hter', hter)
            metric_monitor.update('apcer', apcer)
            metric_monitor.update('bpcer', bpcer)
            metric_monitor.update('acer', acer)
            stream.set_description(
                "mode: {epoch}.  {metric_monitor}".format(
                    epoch="test",
                    metric_monitor=metric_monitor)
            )
    # print(list)
    return metric_monitor.metrics['Accuracy']["avg"], metric_monitor.metrics['F1']["avg"], \
        metric_monitor.metrics['Recall']["avg"], metric_monitor.metrics['eer']["avg"], metric_monitor.metrics['hter'][
        'avg'], \
        metric_monitor.metrics['apcer']['avg'], metric_monitor.metrics['bpcer']['avg'], metric_monitor.metrics['acer'][
        'avg']


def show_heatmaps(title, x_labels, y_labels, harvest, save_name):
    # 这里是创建一个画布
    fig, ax = plt.subplots()
    # cmap https://blog.csdn.net/ztf312/article/details/102474190
    im = ax.imshow(harvest, cmap="OrRd")
    # 这里是修改标签
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(y_labels)))
    ax.set_yticks(np.arange(len(x_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(y_labels)
    ax.set_yticklabels(x_labels)

    # 因为x轴的标签太长了，需要旋转一下，更加好看
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 添加每个热力块的具体数值
    # Loop over data dimensions and create text annotations.
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            text = ax.text(j, i, round(harvest[i, j], 2),
                           ha="center", va="center", color="black")
    ax.set_xlabel("Predict label")
    ax.set_ylabel("Actual label")
    ax.set_title(title)
    fig.tight_layout()
    plt.colorbar(im)
    plt.savefig(save_name, dpi=100)                         
    plt.show()


if __name__ == '__main__':
    test_transform = transforms.Compose(
        [transforms.Resize((params['img_size'], params['img_size'])),  # cannot 224, must (224, 224)
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    # test_dataset = Casia_fasd(params["test_dir"], is_training=False, is_testing=True, transform=test_transform)
    # test_dataset = MSU_MFSD(params['test_dir'], is_train=False, is_test=True, transform=test_transform)
    test_dataset = ReplayAttack(params['test_dir'], transform=test_transform)
    # test_dataset = OULU_NPU1(params["test_dir"], test_transform, protocol='1', testing=True)    ##################  todo 数据集
    test_num = len(test_dataset)
    print("using {} images for test".format(test_num))
    test_loader = DataLoader(  # 按照批次加载训练集
        test_dataset, batch_size=params['batch_size'], shuffle=True,
        num_workers=params['num_workers'], pin_memory=True,
    )
    # 加载模型
    # model = ResNet18(classes_num=2)  # 加载模型结构，加载模型结构过程中pretrained设置为False即可。
    # model = ResNet34(classes_num=2)
    # model = CDCNpp()
    model = ResNet18()
    weights = torch.load(model_path)
    model.load_state_dict(weights)
    model.eval()
    model.to(device)
    # 指标上的测试结果包含三个方面，分别是acc f1 和 recall, 除此之外，应该还有相应的热力图输出，整体会比较好看一些。
    acc, f1, recall, eer, hter, apcer, bpcer, acer = test(test_loader, model, params)
    time_format = "%Y-%m-%d_%H-%M-%S"
    current_time = time.time()
    formatted_time = time.strftime(time_format, time.localtime(current_time))
    print("测试结果：")
    print(
        f"acc: {acc}, F1: {f1}, recall: {recall}, eer:{eer}, hter:{hter}, 'apcer':{apcer}, 'bpcer':{bpcer}, 'acer, {acer}")
    fpath = r"D:\BaiduNetdiskDownload\CDCN-master\CDCN-master\record.txt"
    with open(fpath, "a") as f:
        f.write('\n')
        f.write(
                "R-C-chidu\n"
                f'Data:{data_path}\n'
                f"RunTime: {formatted_time}\n"
                f"weights:{model_path}\n"
                f"image_size:{params['img_size']}\n"
                f"batch_size:{params['batch_size']}\n")
        f.write(
            f"acc: {acc:.5f}, F1: {f1:.5f}, recall: {recall:.5f}, eer:{eer:.5f}, hter:{hter:.5f}, 'apcer':{apcer:.5f}, 'bpcer':{bpcer:.5f}, 'acer, {acer:.5f}")
        f.write('\n')
    print("测试完成，heatmap保存在{}下".format("record"))
