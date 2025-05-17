"""
@Project ：machine-learning 
@File    ：04-nn_demo01.py
@IDE     ：PyCharm 
@Author  ：jinke
@Date    ：2025/4/17 20:31 
"""
import torch
from sklearn import preprocessing
import numpy as np

feature = torch.rand(size=(365, 14), dtype=torch.float, requires_grad=False)
labels = torch.rand(size=(365, 1), dtype=torch.float, requires_grad=True)

# 标准化特征
feature_std = preprocessing.StandardScaler().fit_transform(feature.numpy())
x = torch.tensor(feature_std, requires_grad=True)

"""
稍微简单点的模型构建
"""
input_size = feature.shape[1]
hidden_size = 128
output_size = 1
batch_size = 16

my_nn = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size, output_size)
)
cost = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(my_nn.parameters(), lr=0.001)

# 训练网络
losses = []
for i in range(1000):
    batch_loss = []
    # MINI-Batch方法来进行训练
    for start in range(0, len(feature), batch_size):
        end = start + batch_size if start + batch_size < len(feature) else len(feature)
        xx = torch.tensor(feature[start:end], dtype=torch.float, requires_grad=True)
        yy = torch.tensor(labels[start:end], dtype=torch.float, requires_grad=True)
        prediction = my_nn(xx)
        loss = cost(prediction, yy)
        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        optimizer.step()
        batch_loss.append(loss.data.numpy())

    # 打印损失
    if i % 100 == 0:
        losses.append(np.mean(batch_loss))
        print(i, np.mean(batch_loss))

x = torch.tensor(feature, dtype=torch.float)
predict = my_nn(x).data.numpy()
print(predict)
