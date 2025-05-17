"""
@Project ：machine-learning 
@File    ：05-cnn-02.py
@IDE     ：PyCharm 
@Author  ：jinke
@Date    ：2025/4/20 20:10
整体逻辑：锁住所有层除了FC层-》保存最好的模型-》加载最好的模型，解锁所有参数-》重新训练
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torchvision import transforms, datasets

from cnn_utils import initialize_model, process_image, imshow, im_convert

data_transforms = {
    'train': transforms.Compose([transforms.RandomRotation(45),  # 随机旋转，-45到45度之间随机选
                                 transforms.CenterCrop(224),  # 从中心开始裁剪，留下224*224的。（随机裁剪得到的数据更多）
                                 transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率去翻转，0.5就是50%翻转，50%不翻转
                                 transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
                                 transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
                                 # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
                                 transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度率，3通道就是R=G=B
                                 transforms.ToTensor(),  # 转成tensor的格式
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差（拿人家算好的）
                                 ]),
    'valid': transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 要和训练集保持一致的标准化操作
                                 ]),
}

# pytorch下载，训练集数量不对
# train_dataset = datasets.Flowers102('data/Flowers102/', download=True, split='train',
#                                     transform=data_transforms['train'])
# val_dataset = datasets.Flowers102('data/Flowers102/', download=True, split='val',
#                                   transform=data_transforms['valid'])
# test_dataset = datasets.Flowers102('data/Flowers102/', download=True, split='test',
#                                    transform=data_transforms['valid'])
#
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# 可视化数据集图像
# import matplotlib.pyplot as plt
# n = 10  # 展示10张图像
# plt.figure(figsize=(10, 5))
# for i in range(n):
#     image, labels = train_dataset[i]
#     plt.subplot(2, 5, i+1)
#     plt.imshow(image.permute(1, 2, 0))
#     plt.title(f'Label: {labels}')
# plt.show()

# 路径设置
data_dir = os.path.join('data', 'flower_data')
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')

batch_size = 8  # 减小压力
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
               for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
print(f'dataset_sizes: {dataset_sizes}')
# 类别
class_names = image_datasets['train'].classes
print(f'train class_names: {class_names}')

# 类别和名称的映射关系
with open(os.path.join(data_dir, 'cat_to_name.json'), 'r') as f:
    cat_to_name = json.load(f)

# 展示下数据
# fig = plt.figure(figsize=(20, 12))
# columns = 4
# rows = 2
# dataiter = iter(dataloaders['valid'])
# inputs, classes = dataiter.__next__()
#
# for idx in range(columns * rows):
#     ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
#     # 利用json文件将其对应花的类型打印在图片中
#     ax.set_title(cat_to_name[str(int(class_names[classes[idx]]))])
#     plt.imshow(im_convert(inputs[idx]))
# plt.show()

# 是否用GPU进行训练
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.   Training on CPU ...')
else:
    print('CUDA is available! Training on GPU ...')
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

# 设置模型名字、输出分类数
model_name = 'resnet'
# 是否用人家训练好的特征
feature_extract = True
model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)
print(model_ft)

# GPU 计算
model_ft = model_ft.to(device)

# 模型保存, checkpoints 保存是已经训练好的模型，以后使用可以直接读取
filename = 'checkpoint.pth'

# 是否训练所有层
params_to_update = model_ft.parameters() if not feature_extract else []
# 打印出需要训练的层
print("Params to learn:")
if feature_extract:
    # model_ft 不需要训练的层已经冻住了
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            print("\t", name)

# 优化器设置
optimizer_ft = optim.Adam(params_to_update, lr=1e-2)
# 学习率衰减策略，学习率每7个epoch衰减为原来的1/10，一般都要加
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# 最后一层使用LogSoftmax(), 故不能使用nn.CrossEntropyLoss()来计算
# nn.CrossEntropyLoss，内部会自动对输入应用 softmax 操作，然后计算负对数似然损失。
criterion = nn.NLLLoss()
# 若太慢，把epoch调低，迭代50次可能好些
# 训练时，损失是否下降，准确是否有上升；验证与训练差距大吗？若差距大，就是过拟合
# model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = \
#     train_model(model_ft, dataloaders, criterion, optimizer_ft, scheduler, num_epochs=5,
#                 is_inception=(model_name == "inception"))

"""
训练所有层，需要显卡支持
"""
# 将全部网络解锁进行训练
# for param in model_ft.parameters():
#     param.requires_grad = True
#
# # 再继续训练所有的参数，学习率调小一点
# optimizer = optim.Adam(params_to_update, lr=1e-4)
# scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
# criterion = nn.NLLLoss()
#
# # load checkpoint
# checkpoint = torch.load(filename)
# best_acc = checkpoint['best_acc']
# model_ft.load_state_dict(checkpoint['state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer'])
# model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = \
#     train_model(model_ft, dataloaders, criterion, optimizer, scheduler, num_epochs=2,
#                 is_inception=(model_name == "inception"))

"""
加载已经训练的模型
"""
model_val, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)

# GPU 模式
model_val = model_val.to(device)  # 扔到GPU中

# 保存文件的名字
filename = 'checkpoint.pth'

# 加载模型
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_val.load_state_dict(checkpoint['state_dict'])
image_path = r'data/flower_data/valid/3/image_06621.jpg'
img = process_image(image_path)
imshow(img)
plt.show()

"""
推理
"""
# 得到一个batch的测试数据
dataiter = iter(dataloaders['valid'])
images, labels = dataiter.__next__()

model_val.eval()

if train_on_gpu:
    # 前向传播跑一次会得到output
    output = model_val(images.cuda())
else:
    output = model_val(images)

# batch 中有8个数据，每个数据分为102个结果值， 每个结果是当前的一个概率值
print(output.shape)

# 计算得到最大概率
_, preds_tensor = torch.max(output, 1)

# 将秩为1的数组转为 1 维张量
preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(
    preds_tensor.cpu().numpy())

# 展示预测结果
fig = plt.figure(figsize=(20, 20))
columns = 4
rows = 2

for idx in range(columns * rows):
    ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
    plt.imshow(im_convert(images[idx]))
    # 绿色的表示预测是对的，红色表示预测错了
    ax.set_title("{} ({})".format(cat_to_name[str(preds[idx])], cat_to_name[str(labels[idx].item())]),
                 color=("green" if cat_to_name[str(preds[idx])] == cat_to_name[str(labels[idx].item())] else "red"))
plt.show()
