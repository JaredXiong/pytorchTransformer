"""
损失函数模块 - 支持多种损失函数

新增：
- AntiSmoothingLoss：MSE + 方差下界 + 一阶差分下界，配合 warmup 抑制均值回归
- mse_antismooth 损失类型：在 get_loss_function 中注册
"""
import torch
import torch.nn as nn


def get_loss_function(loss_type: str = 'huber', **kwargs):
    """获取损失函数

    Args:
        loss_type: 损失类型 ('mse', 'huber', 'mae', 'smooth_l1', 'mse_antismooth')
        **kwargs: 损失函数参数
            - mse_antismooth 专用：lambda_var, lambda_diff, tau_var, tau_diff,
              lambda_warmup_epochs, current_epoch
    Returns:
        损失函数实例
    """
    if loss_type == 'mse_antismooth':
        return AntiSmoothingLoss(
            lambda_var=kwargs.get('lambda_var', 0.1),
            lambda_diff=kwargs.get('lambda_diff', 0.05),
            tau_var=kwargs.get('tau_var', 0.5),
            tau_diff=kwargs.get('tau_diff', 0.5),
            warmup_epochs=kwargs.get('lambda_warmup_epochs', 20),
        )

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


class AntiSmoothingLoss(nn.Module):
    """反平滑复合损失 = MSE + (方差下界惩罚) + (一阶差分下界惩罚)。

    用于缓解时间序列模型的均值回归 / 过度平滑问题：
    - L_main = MSE(pred, target)
    - L_var  = ReLU(tau_var  - std(pred, axis=time) / (std(target, axis=time) + eps)).mean()
    - L_diff = ReLU(tau_diff - std(diff(pred, axis=time)) / (std(diff(target, axis=time)) + eps)).mean()
    - 总损失 = L_main + warmup * (lambda_var * L_var + lambda_diff * L_diff)

    `warmup = min(epoch / warmup_epochs, 1.0)`，避免前若干 epoch 小 batch 触发 std≈0 数值不稳定。
    若 `warmup_epochs <= 0` 或 `epoch=None`，warmup 退化为 1.0（立即生效）。
    """

    def __init__(self, lambda_var: float = 0.1, lambda_diff: float = 0.05,
                 tau_var: float = 0.5, tau_diff: float = 0.5,
                 warmup_epochs: int = 20):
        super().__init__()
        self.lambda_var = lambda_var
        self.lambda_diff = lambda_diff
        self.tau_var = tau_var
        self.tau_diff = tau_diff
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.current_epoch: int = 0
        self._main_loss = nn.MSELoss()

    def set_epoch(self, epoch: int) -> None:
        """由 trainer 在每个 epoch 开头调用，更新 warmup 进度。"""
        self.current_epoch = max(0, int(epoch))

    def _warmup_factor(self) -> float:
        if self.warmup_epochs <= 0:
            return 1.0
        return min(self.current_epoch / self.warmup_epochs, 1.0)

    @staticmethod
    def _per_feature_std_ratio(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """沿时间维度（dim=1）按特征计算 std(pred)/std(target)。"""
        # pred/target: (B, T, F)
        pred_std = pred.std(dim=1)              # (B, F)
        target_std = target.std(dim=1)          # (B, F)
        return pred_std / (target_std + eps)    # (B, F)

    @staticmethod
    def _per_feature_diff_std_ratio(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """沿时间维度计算一阶差分的 std 比。"""
        pred_diff = pred[:, 1:] - pred[:, :-1]   # (B, T-1, F)
        target_diff = target[:, 1:] - target[:, :-1]
        pred_std = pred_diff.std(dim=1)          # (B, F)
        target_std = target_diff.std(dim=1)
        return pred_std / (target_std + eps)     # (B, F)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 主损失：MSE（对所有时间步 + 特征求平均）
        l_main = self._main_loss(pred, target)

        # 当 batch 太小或时间步 < 2 时，差分项无意义，跳过
        if pred.size(1) < 2 or target.size(1) < 2:
            l_var = torch.zeros((), device=pred.device, dtype=pred.dtype)
            l_diff = torch.zeros((), device=pred.device, dtype=pred.dtype)
        else:
            var_ratio = self._per_feature_std_ratio(pred, target)         # (B, F)
            l_var = torch.relu(self.tau_var - var_ratio).mean()

            diff_ratio = self._per_feature_diff_std_ratio(pred, target)   # (B, F)
            l_diff = torch.relu(self.tau_diff - diff_ratio).mean()

        warmup = self._warmup_factor()
        l_extra = self.lambda_var * l_var + self.lambda_diff * l_diff
        return l_main + warmup * l_extra
