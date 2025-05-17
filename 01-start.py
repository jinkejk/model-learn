"""
@Project ：machine-learning 
@File    ：test.py
@IDE     ：PyCharm 
@Author  ：jinke
@Date    ：2025/4/14 20:34
"""
import torch
import numpy as np

print(torch.__version__)
print(torch.cuda.is_available())
# x = torch.zeros(5, 3, dtype=torch.int16)
x = torch.randn(4, 4)
# y = x.view(-1, 8)  # -1 自动计算
print(x)

# tensor 和 numpy 转换
# z = x.numpy()
# a = np.ones(5)
# b = torch.from_numpy(a)

# 自动求导 
x.requires_grad = True
b = torch.randn(4, 4, requires_grad=True)
w = torch.randn(4, 4, requires_grad=True)

y = w * x
z = (y + b).sum()
print(x.requires_grad, b.requires_grad, y.requires_grad)
print(x.is_leaf, b.is_leaf, z.is_leaf)
z.backward()
print(w.grad)
print(b.grad)

"""
@ 和 *代表矩阵的两种相乘方式：@表示常规的数学上定义的矩阵相乘；*表示两个矩阵对应位置处的两个元素相乘。
x.dot(y): 向量乘积,x，y均为一维向量。
*和torch.mul()等同:表示相同shape矩阵点乘，即对应位置相乘，得到矩阵有相同的shape。
@和torch.mm(a, b)等同：正常矩阵相乘，要求a的列数与b的行数相同。
torch.mv(X, w0):是矩阵和向量相乘.第一个参数是矩阵，第二个参数只能是一维向量,等价于X乘以w0的转置
Y.t():矩阵Y的转置。
"""
