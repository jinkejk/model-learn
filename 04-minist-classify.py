"""
@Project ：machine-learning 
@File    ：04-nn_demo02.py
@IDE     ：PyCharm 
@Author  ：jinke
@Date    ：2025/4/17 21:02 
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torch import nn, optim

# 下载MNIST数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST('data/MNIST_data/', download=True, train=True, transform=transform)
test_dataset = datasets.MNIST('data/MNIST_data/', download=True, train=False, transform=transform)

# 可视化数据集图像
# import matplotlib.pyplot as plt
# n = 10  # 展示10张图像
# plt.figure(figsize=(10, 5))
# for i in range(n):
#     images, labels = train_set[i]
#     plt.subplot(2, 5, i+1)
#     plt.imshow(images[0].view(28, 28), cmap='gray')
#     plt.title(f'Label: {labels}')
# plt.show()
# print(train_dataset.data.shape)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        # https://www.zhihu.com/question/358069078 softmax和log softmax差别，主要是防止溢出
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 训练模型
epochs = 30
for epoch in range(epochs):  # 训练10个epochs
    sum_loss = 0.0
    train_correct = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        net.train()  # 使用batch normalization和dropout
        net.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, id = torch.max(outputs.data, 1)
        sum_loss += loss.data
        train_correct += torch.sum(id == labels.data)
    print('[%d,%d] loss:%.03f' % (epoch + 1, epochs, sum_loss / len(train_loader)))
    print('correct:%.03f%%' % (100 * train_correct / len(train_dataset)))

torch.save(net.state_dict(), 'model.minist')
net.load_state_dict(torch.load('model.minist'))

# 测试
test_correct = 0
net.eval()  # 测试时不使用batch normalization和dropout
for i, data in enumerate(test_loader):
    inputs, labels = data
    outputs = net(inputs)
    _, id = torch.max(outputs.data, 1)
    test_correct += torch.sum(id == labels.data)

print('correct:%.03f%%' % (100 * test_correct / len(test_dataset)))


"""
自定义dataset
"""
from torch.utils.data import Dataset


class TensorDataset(Dataset):
    """
    TensorDataset继承Dataset, 重载了__init__(), __getitem__(), __len__()
    实现将一组Tensor数据对封装成Tensor数据集
    能够通过index得到数据集的数据，能够通过len，得到数据集大小
    """
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

# 生成数据
data_tensor = torch.randn(4, 3)
target_tensor = torch.rand(4)

# 将数据封装成Dataset
tensor_dataset = TensorDataset(data_tensor, target_tensor)

# 可使用索引调用数据
print(tensor_dataset[1])
# 输出：(tensor([-1.0351, -0.1004,  0.9168]), tensor(0.4977))

# 获取数据集大小
print(len(tensor_dataset))
# 输出：4
