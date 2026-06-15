"""
损失函数模块 - 支持多种损失函数
"""
import torch
import torch.nn as nn


def get_loss_function(loss_type: str = 'huber', **kwargs):
    """获取损失函数

    Args:
        loss_type: 损失类型 ('mse', 'huber', 'mae', 'smooth_l1')
        **kwargs: 损失函数参数
    Returns:
        损失函数实例
    """
    loss_functions = {
        'mse': nn.MSELoss(),
        'huber': nn.HuberLoss(delta=kwargs.get('delta', 1.0)),
        'mae': nn.L1Loss(),
        'smooth_l1': nn.SmoothL1Loss(beta=kwargs.get('beta', 1.0)),
    }

    if loss_type not in loss_functions:
        raise ValueError(f"不支持的损失函数类型: {loss_type}，可选: {list(loss_functions.keys())}")

    return loss_functions[loss_type]


class WeightedMSELoss(nn.Module):
    """加权 MSE 损失 - 对关键指标加权"""

    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or [1.0] * 9

    def forward(self, pred, target):
        weights = torch.tensor(self.weights, device=pred.device)
        squared_error = (pred - target) ** 2
        weighted_se = squared_error * weights.unsqueeze(0).unsqueeze(0)
        return weighted_se.mean()
