"""
@Project ：machine-learning 
@File    ：05-classify.py
@IDE     ：PyCharm 
@Author  ：jinke
@Date    ：2025/4/20 08:56 
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST('data/MNIST_data/', download=True, train=True, transform=transform)
test_dataset = datasets.MNIST('data/MNIST_data/', download=True, train=False, transform=transform)

input_size = 28
num_classes = 10
num_epochs = 5
batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            # 输出16*28*28，使用16个卷积核，卷积核大小为5*5，步长为1，padding=2
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            # 输出32*14*14
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # 输出32*7*7
            nn.MaxPool2d(kernel_size=2)
        )
        # 全链接
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # 将输出reshape成(batch_size, 32*7*7)
        x = x.view(x.size(0), -1)
        return self.out(x)


def accuracy(predictions, labels):
    # 输出和标签的shape一致，且元素为0或1
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights


net = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(num_epochs):
    sum_loss = 0.0
    train_correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        net.train()
        output = net(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.data
        right = accuracy(output, target)
        train_correct += right
    print('[%d,%d] loss:%.03f' % (epoch + 1, num_epochs, sum_loss / len(train_loader)))
    print('train correct:%.03f%%' % (100 * train_correct / len(train_dataset)))

# 测试
net.eval()
val_rights = 0
for (data, target) in test_loader:
    output = net(data)
    right = accuracy(output, target)
    val_rights += right

# 计算准确率
print(val_rights)
print('测试集正确率：%.03f%%' % (100. * val_rights / len(test_dataset)))
