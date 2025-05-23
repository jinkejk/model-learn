"""
@Project ：machine-learning 
@File    ：lora.py
@IDE     ：PyCharm 
@Author  ：jinke
@Date    ：2025/5/22 09:30 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        """
        in_dim: the input dimension of the layer we want to modify using LoRA
        out_dim: the respective output dimension of that layer
        rank: a hyperparameter that controls the inner dimension of the matrices A and B
        alpha: a scaling hyperparameter applied to the output of the low-rank adaptation
        """
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        # 我们从一个随机分布中用很小的值初始化了 A，用零初始化了 B,
        # A 的分布的标准差是由秩的平方根决定的(这个选择确保了 A 中的初始值不会太大)。
        # 在训练开始时，在通过反向传播更新 A 和 B 之前，LoRALayer 不会影响原始权重，因为如果 B = 0，则 AB = 0。
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        # 这个因素决定了 LoRA 层引入的对模型现有权重的更改的大小，
        # 较高的 alpha 值意味着对模型行为的较大调整，而较低的值会导致更微小的变化。
        self.alpha = alpha

    def forward(self, x):
        return self.alpha * (x @ self.A @ self.B)


# 在代码中，当通过修改现有的 PyTorch 模型来实现 LoRA 时，实现这种线性层的 LoRA 修改的一个简单方法是
# 用 LinearWithLoRA 层替换每个线性层，该线性层结合了我们之前的 LoRALayer 实现:
class LinearWithLoRA(nn.Module):

    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


# LinearWithLoRAMerged 计算下面等式的左边，LinearWithLoRA 计算等式的右边，两个函数等价
# x.(W+A.B) = x.W + x.A.B
class LinearWithLoRAMerged(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        lora = self.lora.A @ self.lora.B  # Combine LoRA matrices
        # Then combine LoRA with orig. weights
        combined_weight = self.linear.weight + self.lora.alpha * lora.T
        return F.linear(x, combined_weight, self.linear.bias)


device = torch.device('mps' if torch.mps.is_available() else 'cpu')
# device = torch.device('cpu')
print(f'use device {device}')

torch.manual_seed(123)
layer = nn.Linear(10, 2).to(device)
x = torch.randn((1, 10)).to(device)

print("Original output:", layer(x))

layer_lora_1 = LinearWithLoRA(layer, rank=2, alpha=4).to(device)
print("LoRA output:", layer_lora_1(x))

layer_lora_2 = LinearWithLoRAMerged(layer, rank=2, alpha=4).to(device)
print("LoRA output:", layer_lora_2(x))
