"""
@Project ：machine-learning 
@File    ：tensor.py
@IDE     ：PyCharm 
@Author  ：jinke
@Date    ：2025/4/16 20:14 
"""
import torch

from torch import tensor

# scalar
x = tensor(42.)
print(x.item())

# vector
v = tensor([1.5, 0.5, 3.0])
print(v.dim())

# matrix
m = tensor([[1., 2.], [3., 4.]])
print(m.matmul(m))
print(m * m)

model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
model.eval()
