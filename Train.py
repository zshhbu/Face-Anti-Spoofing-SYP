import sys
from torchutils import *
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm
import time

# from Model.r1esnet18 import ResNet18
# from Model.resnet_1023 import ResNet18
# from Model.resnet_1120 import ResNet18   # 两个模块都加
# from Model.resnet18 import ResNet18  # 原始resnet18
# from Model.resnet18_att import ResNet18 # 注意力
# from Model.resnet18_chidu import ResNet18   # 多尺度
# from Model.resnet_maxpool import ResNet18
# from Model.resnet_chiduxiaorong import ResNet18
from Model.resnet_unpooling_32 import ResNet18
import warnings

warnings.filterwarnings("ignore")
# from renet34 import ResNet34
path = r"D:\BaiduNetdiskDownload\MSU-MFSD\scene01\path.txt"  # mus_fusd标签文件
# train_path = r"D:\BaiduNetdiskDownload\replayattack\train\train.txt"  # rp 训练标签文件
# val_path = r"D:\BaiduNetdiskDownload\replayattack\devel\dev.txt"  # replayattack 测试标签文件
train_path = r"D:\BaiduNetdiskDownload\replayattack\train\train.txt"  # rp 训练标签文件
val_path = r"D:\BaiduNetdiskDownload\replayattack\devel\dev.txt"  # replayattack 测试标签文件
oulu_train_path = r"D:\BaiduNetdiskDownload\oulu_npu\train"
oulu_dev_path = r"D:\BaiduNetdiskDownload\oulu_npu\dev"
oulu_ycbcr_train = r"D:\BaiduNetdiskDownload\oulu_npu\Ycbcr_train"
oulu_ycbcr_dev = r"D:\BaiduNetdiskDownload\oulu_npu\Ycbcr_dev"
casia_path = r"D:\BaiduNetdiskDownload\cssia_fasd\train_img\train_img\color"
save_path = os.path.join('checkpoints', "CA_16_5-5")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
param = {
    'train_path': train_path,
    'val_path': val_path,
    'oulu_train_path': oulu_train_path,
    'oulu_dev_path': oulu_dev_path,
    'oulu_ycbcr_train': oulu_ycbcr_train,
    'oulu_ycbcr_dev': oulu_ycbcr_dev,
    'casia_rgbpath': casia_path,
    'save_dir': save_path,
    'path': path,
    'batch_size': 32,
    'epoch': 100,
    'img_size': 64,
    'number_worker': 0,
    'device': device,
    'model': "resnet18",
    'lr': 0.0001,
    'weight_decay': 1e-5  # 学习率衰减

}
# 获取当前时间戳
current_time = time.time()
# 将时间戳转换为可读的日期时间格式
time_format = "%Y-%m-%d_%H-%M-%S"
formatted_time = time.strftime(time_format, time.localtime(current_time))
folder_name = formatted_time
file_path = os.path.join('logs', param['model'] + folder_name + ".txt")


def main():
    # write param in file
    with open(file_path, "a") as file:
        selected_keys = ['batch_size', 'epoch', 'model', 'img_size', 'save_dir']
        for key in selected_keys:
            value = param[key]
            file.write(f"{key}: {value}\n")
    print("using {} device.".format(device))
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(param['img_size']),
                                     # transforms.RandomHorizontalFlip(p=0.5),
                                     # transforms.RandomRotation((-5, 5)),
                                     # transforms.RandomAutocontrast(p=0.2),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
        "val": transforms.Compose(
            [transforms.Resize((param['img_size'], param['img_size'])),  # cannot 224, must (224, 224)
             transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])}
    # train_dataset = MSU_MFSD(param['path'], transform=data_transform["train"], is_train=True, is_test=False)  # 加载训练集, train_ratio=0.7,is_training=True
    # valid_dataset = MSU_MFSD(param['path'], transform=data_transform["val"], is_train=False, is_test=False)  # 加载验证集
    # train_dataset = ReplayAttack(param['train_path'], transform=data_transform['train'])
    # valid_dataset = ReplayAttack(param['val_path'], transform=data_transform["val"])
    train_dataset = Casia_fasd(param['casia_rgbpath'], is_training=True, is_testing=False, train_ratio=0.7, transform=data_transform["train"])  # 加载训练集, train_ratio=0.7,is_training=True
    valid_dataset = Casia_fasd(param['casia_rgbpath'], is_training=False, is_testing=False, train_ratio=0.7, transform=data_transform["val"])  # 加载验证集

    # train_dataset = OULU_NPU1(path=param['oulu_train_path'],
    #                          transform=data_transform['train'],
    #                          protocol='1',
    #                          testing=False)
    # valid_dataset = OULU_NPU1(path=param['oulu_dev_path'],
    #                          transform=data_transform['val'],
    #                          protocol='1',
    #                          testing=False)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=param['batch_size'],
                                               shuffle=True,
                                               num_workers=0,
                                               drop_last=True)

    # train image number
    train_num = len(train_dataset)
    val_num = len(valid_dataset)
    validate_loader = torch.utils.data.DataLoader(valid_dataset,
                                                  batch_size=64, shuffle=True,
                                                  num_workers=0)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    with open(file_path, "a") as file:
        file.write("using {} images for training, {} images for validation.\n".format(train_num, val_num))
    net = ResNet18()
    net.to(device)
    loss_function = nn.BCEWithLogitsLoss()
    # loss_function = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])

    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(param['epoch']):
        # train
        net.train()
        running_loss = 0.0
        train_acc = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            img, label = data
            rgb_image = img.to(device)
            label = [int(l) for l in label]
            target = torch.tensor(label).to(device)
            labels = torch.nn.functional.one_hot(target, num_classes=2).to(torch.float)
            optimizer.zero_grad()
            outputs = net(rgb_image)
            loss = loss_function(outputs, labels.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc = torch.eq(predict_y, target).sum() / param['batch_size']
            loss.backward()
            optimizer.step()
            adjust_learning_rate(optimizer, epoch, param, step, train_steps)
            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} acc:{:.3f}".format(epoch + 1, param['epoch'], loss, acc)
            train_acc += acc
        # Train_loss = running_loss / (step + 1)
        # Train_acc = train_acc / (step + 1)
        # with open(file_path, "a") as file:
        #     file.write("train epoch[{}/{}] loss:{:.3f} acc:{:.3f}\n".format(epoch + 1, param['epoch'], Train_loss, Train_acc))

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                img, label = val_data
                rgb_image = img.to(device)
                label = [int(l) for l in label]
                target = torch.tensor(label).to(device)
                outputs = net(rgb_image)
                # 预测概率
                predict_y = torch.max(outputs, dim=1)[1]
                # # 测量TP,TN,FP,FN
                # pad_meter_val.update(target.cpu().data.numpy(), predict_y.cpu().data.numpy())
                # # 求eer,阈值thr
                # eer, thr = pad_meter_val.get_eer_and_thr()
                # with open('val_threshold.txt', 'a') as file:
                #     file.write(f"{thr:.5f}\n")
                # 用阈值thr求hter,apcer,bpcer
                # hter, apcer, bpcer, acer = pad_meter_val.get_hter_apcer_etal_at_thr(pad_meter_val.threshold)
                # 用阈值thr求acc
                # acc = pad_meter_val.get_accuracy(pad_meter_val.threshold)
                acc += torch.eq(predict_y, target.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' % (epoch + 1, running_loss / train_steps, val_accurate))
        with open(file_path, "a") as file:
            file.write('[epoch %d] train_loss: %.3f  val_accuracy: %.3f\n' % (
                epoch + 1, running_loss / train_steps, val_accurate))
        save_interval = 1
        os.makedirs(param['save_dir'], exist_ok=True)
        if epoch % save_interval == 0:  # epoch > 50 and
            # 构建权重文件名
            weight_filename = f"model_epoch_{epoch}.pth"
            save_path = os.path.join(param['save_dir'], weight_filename)
            # 保存模型权重
            torch.save(net.state_dict(), save_path)
        if val_accurate > best_acc:
            best_acc = val_accurate
            s = os.path.join(param['save_dir'], 'Best.pth')
            torch.save(net.state_dict(), s)

    print('Finished Training')


if __name__ == '__main__':
    main()
