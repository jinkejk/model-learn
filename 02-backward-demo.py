"""
@Project ：machine-learning 
@File    ：backward-demo.py
@IDE     ：PyCharm 
@Author  ：jinke
@Date    ：2025/4/16 10:58 
"""
import torch
import torch.autograd

"""
标量对标量的求导
"""
# 需要反向传播的数值要为浮点型,不可以为整数.
x = torch.tensor([2.0], requires_grad=True)
print("x = ", x)
y = (x ** 3).sum()
print("y = ", y)

# 只有当requires_grad和is_leaf同时为真时,才会计算节点的梯度值.
y.backward()  # 反向传播,求解导数
print("x.grad = ", x.grad)

# .backward()默认计算对计算图叶子节点的导数,中间过程的导数是不计算的.
x = torch.tensor(3.0, requires_grad=True)
y = 2 * x
z = y ** 2
f = z + 2
f.backward()
# 报错
# print(f"x.grad: {x.grad}, y.grad: {y.grad}, z.grad: {z.grad}")

"""
标量对向量求导
"""
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
w = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
b = 10
y = torch.dot(x, w) + b
y.backward()
print(x.grad)

x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
y = x + 1
z = y ** 2
f = torch.mean(z)

f.backward()
print(x.grad)

"""
backward输出向量或矩阵
当输出为向量或矩阵时,需要通过地一个参数gradient来实现.
gradient参数的维度与最终的输出要保持一致,
gradient中每个元素表示为对应的最终输出中相同位置元素所对应的权重.
"""
print('====================================================================')
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
y = x ** 2 + x

gradient1 = torch.tensor([[1., 1., 1.], [1., 1., 1.]])
y.backward(gradient1)
print(x.grad)

x.grad.zero_()
y = x ** 2 + x
gradient2 = torch.tensor([[1., 0.1, 0.01], [1., 1., 1.]])
y.backward(gradient2)
print(x.grad)
