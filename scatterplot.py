import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 读取人脸欺诈数据
path = "D:\BaiduNetdiskDownload\MSU-MFSD\scene01/test.txt"
fraud_data = pd.read_csv("D:\BaiduNetdiskDownload\MSU-MFSD\scene01/test.txt")
def get_label(path):
    labels = []
    with open(path, 'r', encoding='utf-8') as file:
        # 逐行读取文件内容
        for line in file:
            label = ('1' if line[46:50] == "real" else '0')
            labels.append(int(label))
    return labels


# 提取特征和标签
X = fraud_data  # 假设'label'是标签列
y = get_label(path)

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用PCA进行降维
n_components = 2
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# 绘制散点图
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.colorbar(scatter, label='Label')
plt.title('Face Fraud Data - PCA Scatter Plot')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
