"""
@Project ：machine-learning 
@File    ：05-cnn-utils.py
@IDE     ：PyCharm 
@Author  ：jinke
@Date    ：2025/4/21 22:18 
"""
import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models


def im_convert(tensor):
    """数据展示"""
    # 张量必须位于 CPU 上才能调用 numpy()，否则会报错。
    # clone() 和 detach() 得到的数据会和原数据共享指针，因此如果直接修改底层数据，原张量和新张量都会受到影响。
    # 如果需要完全独立的副本（包括数据和梯度信息），可以使用 deepcopy() 或者 tensor.clone().detach().requires_grad_(True)。
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()  # 移除 NumPy 数组中所有大小为 1 的维度
    # transpose是调换位置，之前是换成了（c， h， w），需要重新还原为（h， w， c）
    image = image.transpose(1, 2, 0)  # numpy的transpose支持多个维度转换，pytorch的transpose只支持两个
    # 反正则化（反标准化）
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))

    # 将图像中小于0 的都换成0，大于的都变成1
    image = image.clip(0, 1)

    return image


# 将一些层定义为false，使其不自动更新
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # 选择适合的模型，不同的模型初始化参数不同
    if model_name == "resnet":
        """
        Resnet18
        """
        # 1. 加载与训练网络
        model_ft = models.resnet18(pretrained=use_pretrained)
        # 2. 是否将提取特征的模块冻住，只训练FC层
        set_parameter_requires_grad(model_ft, feature_extract)
        # 3. 获得全连接层输入特征
        num_frts = model_ft.fc.in_features
        # 4. 重新加载全连接层，设置输出102
        model_ft.fc = nn.Sequential(nn.Linear(num_frts, 102),
                                    nn.LogSoftmax(dim=1))  # 默认dim = 0（对列运算），我们将其改为对行运算，且元素和为1
        input_size = 224
    elif model_name == "alexnet":
        """
        Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)

        # 将最后一个特征输出替换 序号为【6】的分类器
        num_frts = model_ft.classifier[6].in_features  # 获得FC层输入
        model_ft.classifier[6] = nn.Linear(num_frts, num_classes)
        input_size = 224
    elif model_name == "vgg":
        """
        VGG11_bn
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_frts = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_frts, num_classes)
        input_size = 224
    elif model_name == "squeezenet":
        """
        Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224
    elif model_name == "densenet":
        """
        Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_frts = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_frts, num_classes)
        input_size = 224
    elif model_name == "inception":
        """
        Inception V3
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)

        num_frts = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_frts, num_classes)

        num_frts = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_frts, num_classes)
        input_size = 299
    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft, input_size


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=10, is_inception=False, filename='checkpoint.pth'):
    since = time.time()
    # 保存最好的准确率
    best_acc = 0
    """
    checkpoint = torch.load(filename)
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.class_to_idx = checkpoint['mapping']
    """
    # 指定用GPU还是CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 下面是为展示做的
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    LRs = [optimizer.param_groups[0]['lr']]
    # 最好的一次存下来
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 训练和验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 训练
            else:
                model.eval()  # 验证

            running_loss = 0.0
            running_corrects = 0

            # 把数据都取个遍
            for inputs, labels in dataloaders[phase]:
                # 下面是将inputs,labels传到GPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零
                optimizer.zero_grad()
                # 只有训练的时候计算和更新梯度
                with torch.set_grad_enabled(phase == 'train'):
                    # if这面不需要计算，可忽略
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:  # resnet执行的是这里
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        # 概率最大的返回preds
                    _, preds = torch.max(outputs, 1)

                    # 训练阶段更新权重
                    if phase == 'train':
                        loss.backward()
                        scheduler.step()

                # 计算损失
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # 打印操作
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 得到最好那次的模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # 模型保存
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    # tate_dict变量存放训练过程中需要学习的权重和偏执系数
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                scheduler.step(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))  # // 表示向下取整
    print('Best val Acc: {:4f}'.format(best_acc))

    # 保存训练完后用最好的一次当做模型最终的结果
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs


def process_image(image_path):
    # 读取测试集数据
    img = Image.open(image_path)
    # Resize, thumbnail方法只能进行比例缩小，所以进行判断
    # 与Resize不同
    # resize()方法中的size参数直接规定了修改后的大小，而thumbnail()方法按比例缩小
    # 而且对象调用方法会直接改变其大小，返回None
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))

    # crop操作， 将图像再次裁剪为 224 * 224
    left_margin = (img.width - 224) / 2  # 取中间的部分
    bottom_margin = (img.height - 224) / 2
    right_margin = left_margin + 224  # 加上图片的长度224，得到全部长度
    top_margin = bottom_margin + 224

    img = img.crop((left_margin, bottom_margin, right_margin, top_margin))

    # 相同预处理的方法
    # 归一化
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    # 注意颜色通道和位置
    img = img.transpose((2, 0, 1))

    return img


def imshow(image, ax=None, title=None):
    """展示数据"""
    if ax is None:
        fig, ax = plt.subplots()

    # 颜色通道进行还原
    image = np.array(image).transpose((1, 2, 0))

    # 预处理还原
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.set_title(title)

    return ax

