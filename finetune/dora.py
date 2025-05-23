"""
@Project ：machine-learning 
@File    ：dora.py
@IDE     ：PyCharm 
@Author  ：jinke
@Date    ：2025/5/22 09:32 
"""
import torch.nn as nn
import torch.nn.functional as F

from finetune.lora import LoRALayer


class LinearWithDoRAMerged(nn.Module):
    """
    一个结合了Linear层和DoRA层的模块。
    该模块通过使用DoRA技术来增强线性层的表示能力。

    参数:
    - linear: 原始的线性层(nn.Linear)实例
    - rank: DoRA层的秩
    - alpha: DoRA层中的缩放参数
    """

    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear  # 原始的线性层
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )  # 初始化LoRA层
        self.m = nn.Parameter(
            self.linear.weight.norm(p=2, dim=0, keepdim=True)
        )  # 初始化一个参数，用于调整新权重的模长

    # Code loosely inspired by
    # https://github.com/catid/dora/blob/main/dora.py
    def forward(self, x):
        """
        前向传播过程。
        参数:
        - x: 输入特征向量

        返回:
        - 经过更新后的线性变换后的结果
        """
        lora = self.lora.A @ self.lora.B  # 计算LoRA的低秩近似
        numerator = self.linear.weight + self.lora.alpha * lora.T
        denominator = numerator.norm(p=2, dim=0, keepdim=True)
        directional_component = numerator / denominator  # normalize，计算方向组件
        new_weight = self.m * directional_component  # 更新权重
        return F.linear(x, new_weight, self.linear.bias)  # 应用新的权重进行线性变换
