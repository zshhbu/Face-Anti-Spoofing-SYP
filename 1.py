#
# #创建一个用于验证集测试的类变量
# pad_meter_val = PADMeter()
# #图像送入模型
# output,_ = model(img)
# #预测概率
# class_output = nn.functional.softmax(output, dim=1)
# #测量TP,TN,FP,FN
# pad_meter_val.update(target.cpu().data.numpy(),class_output.cpu().data.numpy())
# #求eer,阈值thr
# pad_meter_val.get_eer_and_thr()
# #用阈值thr求hter,apcer,bpcer
# pad_meter_val.get_hter_apcer_etal_at_thr(pad_meter_val.threshold)
# #用阈值thr求acc
# pad_meter_val.get_accuracy(pad_meter_val.threshold)
# ####################################################
# #创建一个用于测试集测试的类变量
# pad_meter_test = PADMeter()
# #图像送入模型
# output,_ = model(img)
# #预测概率
# class_output1 = nn.functional.softmax(output, dim=1)
# #得到TP,TN,FP,FN
# pad_meter_test.update(target.cpu().data.numpy(),class_output1.cpu().data.numpy())
# #在验证集得到的阈值下，测量测试集的hter,apcer等指标
# pad_meter_test.get_hter_apcer_etal_at_thr(pad_meter_val.threshold)
# #验证集得到的阈值下，测量测试集的acc指标
# # pad_meter_test.get_accuracy(pad_meter_val.threshold)
# import os
#
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
#
#
# def rgb_to_ycbcr(rgb_img):
#     r, g, b = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]
#
#     y = 0.299 * r + 0.587 * g + 0.114 * b+16
#     cb = -0.1687 * r - 0.3313 * g + 0.5 * b+128
#     cr = 0.5 * r - 0.4187 * g - 0.0813 * b+128
#
#     ycbcr_img = np.stack((y, cb, cr), axis=-1)
#     return ycbcr_img
#
#
# if __name__ == "__main__":
#     path = r"D:\BaiduNetdiskDownload\oulu_npu\test"
#     i = 0
#     for root, dirs, files in os.walk(path):
#         i += 1
#         print(f"Output written to {i}")
#         for file in files:
#             image_path = os.path.join(root, file)
#     # Load RGB image using OpenCV
#             rgb_image = cv2.imread(image_path)
#             # Convert RGB to YCbCr
#             ycbcr_array = rgb_to_ycbcr(rgb_image)
#             out_path =root.replace("test", "Ycbcr_test")
#             os.makedirs(out_path, exist_ok=True)
#             save_path = os.path.join(out_path, file)
#             cv2.imwrite(save_path, ycbcr_array)

    # # Create a figure with subplots
    # plt.figure(figsize=(12, 6))
    #
    # # Original RGB image
    # plt.subplot(1, 2, 1)
    # plt.title("Original RGB Image")
    # plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    #
    # # YCbCr image
    # plt.subplot(1, 2, 2)
    # plt.title("YCbCr Image (Y Component)")
    # plt.imshow(ycbcr_array[:, :, 0], cmap="gray")
    # plt.axis("off")
    #
    # plt.tight_layout()
    # plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch

def generate_colorbar(cmap, ax):
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))

    ax.imshow(gradient, aspect='auto', cmap=cmap)
    color_range = cmap(np.linspace(0, 1, cmap.N))
    ax.set_xticks([])
    ax.set_yticks([0, 1])
    ax.set_yticklabels([])

    # 添加箭头形状
    arrow_properties = dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=1.5)
    arrow = FancyArrowPatch((0.5, 0), (0.5, 1), connectionstyle="arc3,rad=.5", **arrow_properties)
    ax.add_patch(arrow)

    ax.tick_params(left=False, right=False, labelleft=False, labelright=False)

# 蓝色到黄色到绿色的颜色列表
colors = ['#0000FF', '#FFFF00', '#00FF00']

# 创建自定义色阶
custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)

# 绘制图例
fig, ax = plt.subplots(figsize=(6, 1))
generate_colorbar(custom_cmap, ax)
plt.show()

