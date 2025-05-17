"""
@Project ：machine-learning 
@File    ：linear.py
@IDE     ：PyCharm 
@Author  ：jinke
@Date    ：2025/4/15 20:29 
"""
import numpy as np
import torch
import torch.nn as nn


class LinearRegressionModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModule, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


model = LinearRegressionModule(1, 1)

# gpu 计算
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model.to(device)

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)
y_values = [i*2 + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

epochs = 1000
for epoch in range(epochs):
    epoch += 1
    # 转tensor
    inputs = torch.from_numpy(x_train).to(device)
    labels = torch.from_numpy(y_train).to(device)
    # 梯度清零
    optimizer.zero_grad()
    output = model(inputs)
    # 计算损失
    loss = criterion(output, labels)
    # 反向传播
    loss.backward()
    # 更新权重
    optimizer.step()
    if epoch % 50 == 0:
        print(f'epoch {epoch}, loss {loss.item()}')
        print(f'model params: {model.state_dict()}')

predict = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()
print(predict)

torch.save(model.state_dict(), 'model.pk1')
model.load_state_dict(torch.load('model.pk1'))
