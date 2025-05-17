"""
@Project ：machine-learning 
@File    ：nn_demo01.py
@IDE     ：PyCharm 
@Author  ：jinke
@Date    ：2025/4/16 20:56 
"""
import torch
from sklearn import preprocessing

feature = torch.rand(size=(365, 14), dtype=torch.float, requires_grad=False)
y = torch.rand(size=(365, 1), dtype=torch.float, requires_grad=True)

# 标准化特征
feature_std = preprocessing.StandardScaler().fit_transform(feature.numpy())
x = torch.tensor(feature_std, requires_grad=True)

"""
传统复杂实现，单层神经网络
"""
weights = torch.rand(size=(14, 128), dtype=torch.float, requires_grad=True)
biases = torch.rand(128, dtype=torch.float, requires_grad=True)
weights2 = torch.rand(size=(128, 1), dtype=torch.float, requires_grad=True)
biases2 = torch.rand(1, dtype=torch.float, requires_grad=True)

lr = 0.001
losses = []

for i in range(10000):
    # 计算隐层
    hidden = x.mm(weights) + biases
    # 加入激活函数
    hidden = torch.relu(hidden)
    # 预测结果
    predictions = hidden.mm(weights2) + biases2
    # 通计算损失
    loss = torch.mean((predictions - y) ** 2)
    losses.append(loss.data.numpy())
    # 打印损失值
    if i % 100 == 0:
        print(' loss:', loss)
    # 返向传播计算
    loss.backward()

    # .data得到一个tensor的数据信息，其返回的信息与detach()返回的信息是相同的。
    # 具有内存相同，不保存梯度信息的特点。脱离计算图
    weights.data.add_(- lr * weights.grad.data)
    biases.data.add_(- lr * biases.grad.data)
    weights2.data.add_(- lr * weights2.grad.data)
    biases2.data.add(- lr * biases2.grad.data)
    # 清除梯度
    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()
    biases2.grad.data.zero_()
