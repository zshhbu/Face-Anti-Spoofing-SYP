# 导入必要的库
import cv2
import numpy as np
import tensorflow as tf
from predictor import PosPrediction

pos_predictor = PosPrediction(256, 256)
pos_predictor.restore('./Data/net-data/256_256_resfcn256_weight')

cropped_pos = pos_predictor.predict(cropped_img) #网络推断
