"""
Pretrainer:掩码自监督预训练器

任务:在 VMD 拼接后的特征上,随机遮盖部分时间步/特征,
让 backbone 预测被遮盖的 IMF 分量。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tud

logger = logging.getLogger(__name__)


def compute_masked_mse(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float:
    """仅在 mask=1 的位置计算 MSE。"""
    diff = (pred - target) ** 2
    masked = diff * mask
    denom = float(mask.sum())
    if denom < 1e-6:
        return 0.0
    return float(masked.sum() / denom)


class Pretrainer:
    """掩码自监督预训练器(在 unlabeled 段上训练,预测被遮盖的 IMF)"""

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        learning_rate: float = 5e-4,
        weight_decay: float = 3e-4,
        mask_ratio: float = 0.3,
        early_stop_patience: int = 20,
        gradient_clip: float = 1.0,
    ):
        self.model = model
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.mask_ratio = mask_ratio
        self.early_stop_patience = early_stop_patience
        self.gradient_clip = gradient_clip
        self.history: Dict[str, Any] = {'train_losses': []}

    @staticmethod
    def generate_mask(
        shape: tuple,
        ratio: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """生成 0/1 掩码,1=保留,0=遮盖。"""
        rng = np.random.default_rng(seed)
        B, T, K = shape
        keep_prob = 1.0 - ratio
        mask = (rng.random(shape) < keep_prob).astype(np.float32)
        # 保证每个 batch 至少有一个保留位置,避免除零
        for i in range(B):
            if mask[i].sum() == 0:
                mask[i, 0, 0] = 1.0
        return mask

    def fit(
        self,
        X_unlabeled: np.ndarray,
        epochs: int = 60,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """预训练循环。"""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.model.train()
        self.model.to(self.device)

        loader = tud.DataLoader(
            tud.TensorDataset(torch.FloatTensor(X_unlabeled)),
            batch_size=batch_size or min(32, len(X_unlabeled)),
            shuffle=True,
        )

        best_loss = float('inf')
        patience_counter = 0
        losses = []
        vmd_K = getattr(self.model, 'vmd_K', None)
        if vmd_K is None:
            raise ValueError(
                f"Pretrainer 要求 model 必须有 vmd_K 属性,当前 model 类型: {type(self.model).__name__}"
            )

        for epoch in range(epochs):
            epoch_losses = []
            for (x,) in loader:
                x = x.to(self.device)
                B, T, F = x.shape
                mask = torch.FloatTensor(
                    self.generate_mask((B, T, vmd_K), ratio=self.mask_ratio)
                ).to(self.device)
                x_masked = x.clone()
                x_masked[:, :, :vmd_K] = x_masked[:, :, :vmd_K] * mask

                out = self.model(x_masked, mode='pretrain')
                target = x[:, :, :vmd_K]
                diff = (out - target) ** 2
                denom = mask.sum().clamp(min=1.0)
                loss = (diff * mask).sum() / denom

                optimizer.zero_grad()
                loss.backward()
                if self.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                optimizer.step()
                epoch_losses.append(loss.item())

            epoch_loss = float(np.mean(epoch_losses))
            losses.append(epoch_loss)
            logger.info(f"[Pretrain] epoch={epoch+1}/{epochs} loss={epoch_loss:.6f}")

            if epoch_loss < best_loss - 1e-6:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stop_patience:
                    logger.info(f"[Pretrain] Early stop at epoch {epoch+1}")
                    break

        self.history['train_losses'] = losses
        return self.history

    @staticmethod
    def extract_backbone_state(model: nn.Module) -> Dict[str, torch.Tensor]:
        """提取 backbone 权重,丢弃 pretrain_head 与 finetune_head。"""
        backbone_prefixes = (
            'input_proj', 'conv3', 'conv5', 'conv7',
            'bn', 'bilstm', 'attention',
        )
        sd = model.state_dict()
        return {
            k: v for k, v in sd.items()
            if any(k.startswith(p) for p in backbone_prefixes)
        }

    @staticmethod
    def load_backbone_into(
        model: nn.Module,
        backbone_state: Dict[str, torch.Tensor],
    ) -> None:
        """用 backbone_state 加载到 model(strict=False)。"""
        model.load_state_dict(backbone_state, strict=False)
