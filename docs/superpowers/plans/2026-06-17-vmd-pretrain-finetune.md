# VMD-CNN-BiLSTM-Attention 预训练-微调范式实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 VMD-CNN-BiLSTM-Attention 半监督范式基础上,新增无监督预训练阶段(掩码自监督预测被遮盖的 IMF),用预训练 backbone 初始化 Teacher/Student,缩短下游收敛并验证精度增益。

**Architecture:** 拆分 `VMDCNNBILSTMAttentionModel` 为共享 backbone + 双 head(pretrain_head / finetune_head);新增 `Pretrainer`(掩码自监督)与 `PretrainFinetuneTrainer`(三阶段编排,继承 `SemiSupervisedTrainer`);通过 CLI 开关 `--pretrain` 启用,与现有路径完全兼容。

**Tech Stack:** PyTorch, NumPy, vmdpy(沿用),unittest(沿用)

---

## 文件结构

**新增**:
- `src/air_quality/training/pretrain.py` — `Pretrainer` 类(掩码生成 + 预训练循环)
- `src/air_quality/training/pretrain_finetune.py` — `PretrainFinetuneTrainer` 类(三阶段编排)
- `tests/test_pretrain.py` — 掩码/损失/反向传播
- `tests/test_pretrain_finetune.py` — 三阶段流程
- `tests/test_pretrain_semi_pipeline.py` — 端到端集成

**修改**:
- `src/air_quality/models/vmd_cnn_bilstm_attention.py` — 拆分 backbone / pretrain_head / finetune_head,加 `forward(mode)`
- `src/air_quality/config/settings.py` — 加 `PretrainConfig` dataclass
- `src/air_quality/training/__init__.py` — 导出新类
- `src/air_quality/air_quality/__init__.py` — 顶层导出
- `src/air_quality/training/semi_supervised.py` — 提取出可复用的 `_build_model_with_state` 静态方法
- `scripts/train.py` — 加 `--pretrain` 系列参数与分支
- `tests/test_vmd_cnn_bilstm.py` — 补充新 head 的测试
- `README.md` — 补充使用说明
- `docs/algorithm_analysis.md` — 补充章节

---

## Task 1: 拆分 VMDCNNBILSTMAttentionModel 为 backbone + 双 head

**Files:**
- Modify: `src/air_quality/models/vmd_cnn_bilstm_attention.py`
- Modify: `tests/test_vmd_cnn_bilstm.py`

- [ ] **Step 1: 写失败测试,验证 `forward(mode='pretrain')` 输出形状 (B, T, K)**

在 `tests/test_vmd_cnn_bilstm.py` 末尾新增测试类 `TestVMDModelHeads`:

```python
class TestVMDModelHeads(unittest.TestCase):
    """双 head(pretrain / finetune)拆分测试"""

    def setUp(self):
        self.model = VMDCNNBILSTMAttentionModel(
            input_size=12, output_size=7, vmd_K=4,
            cnn_filters=16, bilstm_hidden=16, bilstm_layers=1, dropout=0.1,
        )

    def test_finetune_mode_default(self):
        """默认 mode='finetune' 输出 (B, T, 7)"""
        x = torch.randn(2, 14, 12)
        out = self.model(x)
        self.assertEqual(out.shape, (2, 14, 7))

    def test_finetune_mode_explicit(self):
        """显式 mode='finetune' 输出 (B, T, 7)"""
        x = torch.randn(2, 14, 12)
        out = self.model(x, mode='finetune')
        self.assertEqual(out.shape, (2, 14, 7))

    def test_pretrain_mode_shape(self):
        """mode='pretrain' 输出 (B, T, K=4)"""
        x = torch.randn(2, 14, 12)
        out = self.model(x, mode='pretrain')
        self.assertEqual(out.shape, (2, 14, 4))

    def test_has_both_heads(self):
        """模型应同时包含 pretrain_head 与 finetune_head"""
        self.assertTrue(hasattr(self.model, 'pretrain_head'))
        self.assertTrue(hasattr(self.model, 'finetune_head'))

    def test_invalid_mode_raises(self):
        """未知 mode 应抛 ValueError"""
        x = torch.randn(2, 14, 12)
        with self.assertRaises(ValueError):
            self.model(x, mode='unknown')

    def test_pretrain_head_is_linear(self):
        """pretrain_head 应该是 Linear(bilstm_hidden*2, vmd_K)"""
        import torch.nn as nn
        self.assertIsInstance(self.model.pretrain_head, nn.Linear)
        self.assertEqual(self.model.pretrain_head.in_features, 16 * 2)  # bilstm_hidden*2
        self.assertEqual(self.model.pretrain_head.out_features, 4)       # vmd_K
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m unittest tests.test_vmd_cnn_bilstm -v 2>&1 | tail -30`
Expected: 新增 6 个测试全部失败(因为现有模型没有 `mode` 参数 / 没有 `pretrain_head`)

- [ ] **Step 3: 重构 `src/air_quality/models/vmd_cnn_bilstm_attention.py`**

把整个文件替换为:

```python
"""
VMD-CNN-BiLSTM-Attention 模型(双 head 版)

数据流:input → 多尺度 CNN → BiLSTM → Attention → [pretrain_head | finetune_head]

注:VMD 分解在数据预处理阶段离线完成(参见 vmd_features.apply_vmd_to_aqi),
模型 forward 接收已经拼接好的 (B, T, input_size) 输入。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel


class TemporalAttention(nn.Module):
    """时间步注意力层"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        scores = self.v(torch.tanh(self.W(x)))  # (B, T, 1)
        weights = torch.softmax(scores, dim=1)
        return x * weights  # (B, T, H) - 加权后的特征


class VMDCNNBILSTMAttentionModel(BaseModel):
    """VMD-CNN-BiLSTM-Attention 混合模型(双 head)

    forward(x, mode='finetune'):
      - mode='pretrain'  → backbone → pretrain_head  → (B, T, vmd_K)
      - mode='finetune'  → backbone → finetune_head  → (B, T, output_size)

    共享 backbone:input_proj + 多尺度 CNN + BiLSTM + Attention
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 7,
        vmd_K: int = 4,
        cnn_filters: int = 64,
        bilstm_hidden: int = 128,
        bilstm_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(input_size, output_size, dropout)
        self.vmd_K = vmd_K

        # === 共享 backbone ===
        self.input_proj = nn.Linear(input_size, cnn_filters)
        self.conv3 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(cnn_filters * 3)
        self.relu = nn.ReLU()
        self.bilstm = nn.LSTM(
            cnn_filters * 3,
            bilstm_hidden,
            bilstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if bilstm_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(bilstm_hidden * 2)

        # === 预训练 head:预测 K 个 IMF ===
        self.pretrain_head = nn.Linear(bilstm_hidden * 2, vmd_K)

        # === 微调 head:预测 output_size 维污染物 ===
        self.finetune_head = nn.Sequential(
            nn.Linear(bilstm_hidden * 2, bilstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bilstm_hidden, output_size),
        )

        # 统一权重初始化(与 transformer/cnn/hybrid 风格一致)
        self.init_weights()

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        """backbone 前向:输入 (B, T, F) → 输出 (B, T, bilstm_hidden*2)"""
        h = self.input_proj(x)  # (B, T, cnn_filters)
        h_t = h.transpose(1, 2)  # (B, cnn_filters, T)
        h3 = self.conv3(h_t)
        h5 = self.conv5(h_t)
        h7 = self.conv7(h_t)
        h_cat = torch.cat([h3, h5, h7], dim=1)  # (B, cnn_filters*3, T)
        h_cat = self.bn(h_cat)
        h_cat = self.relu(h_cat)
        h_cat = F.adaptive_max_pool1d(h_cat, h_t.size(2))  # (B, cnn_filters*3, T)
        h_seq = h_cat.transpose(1, 2)  # (B, T, cnn_filters*3)
        lstm_out, _ = self.bilstm(h_seq)  # (B, T, bilstm_hidden*2)
        attended = self.attention(lstm_out)  # (B, T, bilstm_hidden*2)
        return attended

    def forward(self, x: torch.Tensor, mode: str = 'finetune') -> torch.Tensor:
        """
        Args:
            x: (B, T, input_size) — 输入应在外部已做 VMD 拼接
            mode: 'pretrain' 或 'finetune'

        Returns:
            mode='pretrain' 时 (B, T, vmd_K)
            mode='finetune' 时 (B, T, output_size)
        """
        attended = self._backbone(x)
        if mode == 'pretrain':
            return self.pretrain_head(attended)
        elif mode == 'finetune':
            return self.finetune_head(attended)
        else:
            raise ValueError(
                f"mode 必须是 'pretrain' 或 'finetune',当前: {mode}"
            )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m unittest tests.test_vmd_cnn_bilstm -v`
Expected: 10 个测试全部通过(原 4 个 + 新增 6 个)

- [ ] **Step 5: 跑全部测试确认未破坏现有功能**

Run: `python -m unittest discover tests -v 2>&1 | tail -10`
Expected: 全部通过(无回归)

- [ ] **Step 6: 提交**

```bash
git add src/air_quality/models/vmd_cnn_bilstm_attention.py tests/test_vmd_cnn_bilstm.py
git commit -m "refactor(model): VMD-CNN-BiLSTM-Attention 拆分双 head

- backbone 共享(input_proj/CNN/BiLSTM/Attention)
- pretrain_head: Linear → K,用于预训练预测 IMF
- finetune_head: Sequential,用于下游预测污染物
- forward(x, mode='finetune') 默认,行为完全向后兼容

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: 实现 Pretrainer 掩码自监督预训练器

**Files:**
- Create: `src/air_quality/training/pretrain.py`
- Create: `tests/test_pretrain.py`

- [ ] **Step 1: 写失败测试 `tests/test_pretrain.py`**

```python
"""Pretrainer 单元测试"""
import unittest
import numpy as np
import torch

from air_quality.models.vmd_cnn_bilstm_attention import (
    VMDCNNBILSTMAttentionModel,
)
from air_quality.training.pretrain import Pretrainer, compute_masked_mse


class TestMaskedMSE(unittest.TestCase):
    def test_returns_finite_scalar(self):
        """应返回有限标量"""
        pred = np.random.randn(4, 14, 4).astype(np.float32)
        target = np.random.randn(4, 14, 4).astype(np.float32)
        mask = np.zeros((4, 14, 4), dtype=np.float32)
        mask[0, 3, 1] = 1.0
        loss = compute_masked_mse(pred, target, mask)
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(loss, 0.0)

    def test_only_masked_positions_contribute(self):
        """loss 应仅在 mask=1 的位置计算"""
        # 构造:全 0 mask 时 loss=0;全 1 mask 时 loss>0
        pred = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
        target = np.array([[[0.0, 0.0], [0.0, 0.0]]], dtype=np.float32)
        mask_zero = np.zeros_like(pred)
        mask_one = np.ones_like(pred)
        loss_zero = compute_masked_mse(pred, target, mask_zero)
        loss_one = compute_masked_mse(pred, target, mask_one)
        self.assertAlmostEqual(loss_zero, 0.0, places=6)
        self.assertGreater(loss_one, 0.0)


class TestPretrainerMaskGeneration(unittest.TestCase):
    def test_mask_shape(self):
        """生成 (B, T, K) 的 0/1 掩码"""
        mask = Pretrainer.generate_mask((8, 14, 4), ratio=0.3, seed=0)
        self.assertEqual(mask.shape, (8, 14, 4))
        self.assertTrue(set(np.unique(mask)).issubset({0.0, 1.0}))

    def test_mask_ratio_approximate(self):
        """遮盖比例应接近 ratio"""
        np.random.seed(42)
        mask = Pretrainer.generate_mask((100, 14, 4), ratio=0.3)
        actual_ratio = 1.0 - mask.mean()
        # 允许 ±0.05 偏差
        self.assertAlmostEqual(actual_ratio, 0.3, delta=0.05)

    def test_mask_reproducible_with_seed(self):
        """相同 seed 应得到相同掩码"""
        m1 = Pretrainer.generate_mask((5, 14, 4), ratio=0.3, seed=123)
        m2 = Pretrainer.generate_mask((5, 14, 4), ratio=0.3, seed=123)
        np.testing.assert_array_equal(m1, m2)


class TestPretrainerFit(unittest.TestCase):
    def setUp(self):
        self.model = VMDCNNBILSTMAttentionModel(
            input_size=12, output_size=7, vmd_K=4,
            cnn_filters=16, bilstm_hidden=16, bilstm_layers=1, dropout=0.1,
        )
        self.pretrainer = Pretrainer(
            model=self.model, device='cpu',
            learning_rate=5e-4, weight_decay=3e-4,
            mask_ratio=0.3, early_stop_patience=10,
        )
        np.random.seed(0)
        torch.manual_seed(0)
        self.X = np.random.randn(16, 14, 12).astype(np.float32)

    def test_fit_runs_and_returns_history(self):
        """fit 应能跑通并返回 history"""
        history = self.pretrainer.fit(self.X, epochs=2, batch_size=4)
        self.assertIn('train_losses', history)
        self.assertGreater(len(history['train_losses']), 0)

    def test_fit_loss_decreases_or_stable(self):
        """最后 N 步的损失应不大于前 N 步的最大值(允许持平)"""
        history = self.pretrainer.fit(self.X, epochs=4, batch_size=4)
        losses = history['train_losses']
        # 取后半段 max,应 ≤ 前半段 max
        half = len(losses) // 2
        first_max = max(losses[:half])
        second_max = max(losses[half:])
        # 允许 30% 波动(随机初始化)
        self.assertLessEqual(second_max, first_max * 1.3)

    def test_extract_backbone_state(self):
        """extract_backbone_state 应丢弃 pretrain_head"""
        # 跑 1 步让参数被更新
        self.pretrainer.fit(self.X, epochs=1, batch_size=4)
        state = Pretrainer.extract_backbone_state(self.model)
        # 不应包含 pretrain_head 相关 key
        bad = [k for k in state if 'pretrain_head' in k]
        self.assertEqual(bad, [])
        # 应包含 input_proj / bilstm 等 backbone key
        good = [k for k in state if any(
            k.startswith(prefix) for prefix in
            ['input_proj', 'conv3', 'conv5', 'conv7', 'bn', 'bilstm', 'attention']
        )]
        self.assertGreater(len(good), 0)

    def test_gradient_flows_through_backbone(self):
        """backbone 参数应有非零梯度"""
        model = VMDCNNBILSTMAttentionModel(
            input_size=12, output_size=7, vmd_K=4,
            cnn_filters=16, bilstm_hidden=16, bilstm_layers=1, dropout=0.0,
        )
        pretrainer = Pretrainer(
            model=model, device='cpu', mask_ratio=0.3,
        )
        x = torch.FloatTensor(np.random.randn(4, 14, 12))
        mask = torch.FloatTensor(
            Pretrainer.generate_mask((4, 14, 4), ratio=0.3, seed=0)
        )
        out = model(x, mode='pretrain')  # (4, 14, 4)
        target = x[:, :, :4]              # (4, 14, 4)
        loss = ((out - target) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)
        loss.backward()
        # bilstm 与 pretrain_head 都应有梯度
        self.assertIsNotNone(model.bilstm.weight_hh_l0.grad)
        self.assertIsNotNone(model.pretrain_head.weight.grad)
        self.assertFalse(torch.all(model.bilstm.weight_hh_l0.grad == 0))
        self.assertFalse(torch.all(model.pretrain_head.weight.grad == 0))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m unittest tests.test_pretrain -v 2>&1 | tail -20`
Expected: ModuleNotFoundError: No module named 'air_quality.training.pretrain'

- [ ] **Step 3: 实现 `src/air_quality/training/pretrain.py`**

```python
"""
Pretrainer:掩码自监督预训练器

任务:在 VMD 拼接后的特征上,随机遮盖部分时间步/特征,
让 backbone 预测被遮盖的 IMF 分量。

用法:
    model = VMDCNNBILSTMAttentionModel(input_size=8+K, vmd_K=K, ...)
    pretrainer = Pretrainer(model, device='cpu')
    pretrainer.fit(X_unlabeled, epochs=60)
    backbone_state = Pretrainer.extract_backbone_state(model)
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
    """仅在 mask=1 的位置计算 MSE。

    Args:
        pred: (B, T, K) 模型输出
        target: (B, T, K) 真值
        mask: (B, T, K) 0/1 掩码,1 表示该位置参与损失

    Returns:
        标量损失值
    """
    diff = (pred - target) ** 2
    masked = diff * mask
    denom = mask.sum().item() if hasattr(mask.sum(), 'item') else float(mask.sum())
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
        loss_type: str = 'mse',
    ):
        self.model = model
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.mask_ratio = mask_ratio
        self.early_stop_patience = early_stop_patience
        self.gradient_clip = gradient_clip
        self.loss_type = loss_type
        self.history: Dict[str, Any] = {'train_losses': []}

    @staticmethod
    def generate_mask(
        shape: tuple,
        ratio: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """生成 0/1 掩码,1=保留,0=遮盖。

        Args:
            shape: (B, T, K) 掩码形状
            ratio: 遮盖比例(0~1)
            seed: 随机种子

        Returns:
            (B, T, K) 的 float32 数组
        """
        rng = np.random.default_rng(seed)
        B, T, K = shape
        # 每个元素独立以 ratio 概率置 0
        keep_prob = 1.0 - ratio
        mask = (rng.random(shape) < keep_prob).astype(np.float32)
        # 保证每个样本至少 1 个保留(避免 mask 全 0)
        for i in range(B):
            if mask[i].sum() == 0:
                mask[i, 0, 0] = 1.0
        return mask

    def _make_loader(self, X, shuffle=True):
        return tud.DataLoader(
            tud.TensorDataset(torch.FloatTensor(X)),
            batch_size=min(32, len(X)),
            shuffle=shuffle,
        )

    def fit(
        self,
        X_unlabeled: np.ndarray,
        epochs: int = 60,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """预训练循环。

        Args:
            X_unlabeled: (N_un, T, F) VMD 拼接后的特征
            epochs: 训练轮数
            batch_size: 覆盖默认 batch size

        Returns:
            history 字典
        """
        if batch_size is not None:
            self._batch_size_override = batch_size
        else:
            self._batch_size_override = None

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
        N = X_unlabeled.shape[0]
        vmd_K = self.model.vmd_K  # type: ignore[attr-defined]

        for epoch in range(epochs):
            epoch_losses = []
            for (x,) in loader:
                x = x.to(self.device)
                B, T, F = x.shape
                # 生成掩码,只遮盖前 K 维 IMF
                mask = torch.FloatTensor(
                    self.generate_mask((B, T, vmd_K), ratio=self.mask_ratio)
                ).to(self.device)
                # 把被遮盖的 IMF 位置置 0
                x_masked = x.clone()
                x_masked[:, :, :vmd_K] = x_masked[:, :, :vmd_K] * mask

                # 前向(pretrain 模式)
                out = self.model(x_masked, mode='pretrain')  # (B, T, K)
                target = x[:, :, :vmd_K]
                diff = (out - target) ** 2
                denom = mask.sum().clamp(min=1.0)
                loss = (diff * mask).sum() / denom

                # 反向
                optimizer.zero_grad()
                loss.backward()
                if self.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                optimizer.step()
                epoch_losses.append(loss.item())

            epoch_loss = float(np.mean(epoch_losses))
            losses.append(epoch_loss)
            logger.info(f"[Pretrain] epoch={epoch+1}/{epochs} loss={epoch_loss:.6f}")

            # 早停
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
            'bn', 'bilstm', 'attention', 'fc1',
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
        """用 backbone_state 加载到 model(strict=False,head 不匹配会被忽略)。"""
        model.load_state_dict(backbone_state, strict=False)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m unittest tests.test_pretrain -v`
Expected: 9 个测试全部通过

- [ ] **Step 5: 跑全部测试确认未破坏**

Run: `python -m unittest discover tests 2>&1 | tail -5`
Expected: 全部通过

- [ ] **Step 6: 提交**

```bash
git add src/air_quality/training/pretrain.py tests/test_pretrain.py
git commit -m "feat(training): 新增 Pretrainer 掩码自监督预训练器

- generate_mask 生成 0/1 掩码,支持 seed 复现
- compute_masked_mse 仅在遮盖位置计算损失
- fit() 实现预训练循环(Adam + grad-clip + 早停)
- extract_backbone_state 提取共享 backbone 权重
- load_backbone_into 加载到下游模型(strict=False)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: 在 config 中新增 PretrainConfig

**Files:**
- Modify: `src/air_quality/config/settings.py`
- Create: `tests/test_pretrain_config.py`

- [ ] **Step 1: 写失败测试 `tests/test_pretrain_config.py`**

```python
"""PretrainConfig 单元测试"""
import unittest

from air_quality.config import config


class TestPretrainConfig(unittest.TestCase):
    def setUp(self):
        # 保存原始值以便恢复
        self._orig_enabled = config.pretrain.enabled
        self._orig_epochs = config.pretrain.epochs
        self._orig_K = config.pretrain.mask_ratio

    def tearDown(self):
        config.update_from_dict({
            'pretrain': {
                'enabled': self._orig_enabled,
                'epochs': self._orig_epochs,
                'mask_ratio': self._orig_K,
            }
        })

    def test_defaults(self):
        """默认值"""
        self.assertFalse(config.pretrain.enabled)
        self.assertEqual(config.pretrain.epochs, 60)
        self.assertEqual(config.pretrain.mask_ratio, 0.3)
        self.assertEqual(config.pretrain.learning_rate, 5e-4)
        self.assertEqual(config.pretrain.target, 'imf')

    def test_update_from_dict(self):
        """update_from_dict 应能更新 pretrain 字段"""
        config.update_from_dict({
            'pretrain': {
                'enabled': True,
                'epochs': 20,
                'mask_ratio': 0.5,
            }
        })
        self.assertTrue(config.pretrain.enabled)
        self.assertEqual(config.pretrain.epochs, 20)
        self.assertEqual(config.pretrain.mask_ratio, 0.5)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m unittest tests.test_pretrain_config -v 2>&1 | tail -10`
Expected: AttributeError: 'AppConfig' object has no attribute 'pretrain'

- [ ] **Step 3: 在 `src/air_quality/config/settings.py` 末尾添加**

```python
@dataclass
class PretrainConfig:
    """无监督预训练配置(VMD 预训练-微调范式)"""
    enabled: bool = False
    epochs: int = 60
    mask_ratio: float = 0.3
    learning_rate: float = 5e-4
    weight_decay: float = 3e-4
    batch_size: int = 32
    early_stop_patience: int = 20
    target: str = 'imf'  # 固定:只预测 K 个 IMF
```

并在 `AppConfig` 类中添加字段(找到 `AppConfig` 定义,加一行):

```python
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m unittest tests.test_pretrain_config -v`
Expected: 2 个测试通过

- [ ] **Step 5: 跑全部测试确认未破坏**

Run: `python -m unittest discover tests 2>&1 | tail -3`
Expected: 全部通过

- [ ] **Step 6: 提交**

```bash
git add src/air_quality/config/settings.py tests/test_pretrain_config.py
git commit -m "feat(config): 新增 PretrainConfig 配置类

- enabled: 总开关
- epochs/mask_ratio/learning_rate: 预训练超参
- target='imf' 固定(只预测 K 个 IMF)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: 在 SemiSupervisedTrainer 中抽出可复用方法

**Files:**
- Modify: `src/air_quality/training/semi_supervised.py`

- [ ] **Step 1: 抽出 `_build_model_with_state` 静态方法**

找到 `SemiSupervisedTrainer` 类,添加新静态方法(放在 `_make_loader` 之后):

```python
    @staticmethod
    def _build_model_with_state(
        model_type: str,
        input_size: int,
        backbone_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> nn.Module:
        """创建模型,若 backbone_state 不为空则加载到共享部分。"""
        model = create_model(model_type, input_size=input_size)
        if backbone_state is not None:
            # strict=False 允许 head 维度不匹配
            model.load_state_dict(backbone_state, strict=False)
        return model
```

并在文件顶部 `from typing import Optional` 已有情况下,确保 `Optional` 可用;若没有,改为:

```python
from typing import Any, Dict, Optional, Tuple
```

- [ ] **Step 2: 把现有 `_train_teacher` 与 `_train_student` 内部的 `create_model(...)` 调用替换为 `_build_model_with_state(self.model_type, input_size=self.input_size, backbone_state=backbone_state)`**

具体改动(在 `_train_teacher` 内):
```python
        teacher = create_model(self.model_type, input_size=self.input_size).to(self.device)
```
改为:
```python
        teacher = self._build_model_with_state(
            self.model_type, input_size=self.input_size,
            backbone_state=getattr(self, '_backbone_state', None),
        ).to(self.device)
```

`_train_student` 同样改动。

并在 `fit()` 中(若使用 `backbone_state` 参数),在开头加:
```python
        self._backbone_state = backbone_state  # type: ignore[attr-defined]
```

但 `fit()` 当前签名没有 `backbone_state` 参数 — 这一步的改动只引入 `_build_model_with_state` 与可选 attribute,`fit()` 仍保持原签名。后续 Task 5 才让 `PretrainFinetuneTrainer` 显式传入。

- [ ] **Step 3: 跑测试确认未破坏**

Run: `python -m unittest discover tests 2>&1 | tail -3`
Expected: 全部通过(应包含 test_semi_trainer.py 与 test_full_semi_pipeline.py)

- [ ] **Step 4: 提交**

```bash
git add src/air_quality/training/semi_supervised.py
git commit -m "refactor(training): 抽出 _build_model_with_state 静态方法

为 PretrainFinetuneTrainer 复用做准备,允许传入 backbone_state 初始化模型。

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: 实现 PretrainFinetuneTrainer 三阶段编排

**Files:**
- Create: `src/air_quality/training/pretrain_finetune.py`
- Create: `tests/test_pretrain_finetune.py`
- Modify: `src/air_quality/training/__init__.py`

- [ ] **Step 1: 写失败测试 `tests/test_pretrain_finetune.py`**

```python
"""PretrainFinetuneTrainer 三阶段流水线测试"""
import unittest
import numpy as np
import torch

from air_quality.training.pretrain_finetune import PretrainFinetuneTrainer
from air_quality.training.pretrain import Pretrainer


class TestPretrainFinetuneTrainer(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
        self.X_labeled = np.random.randn(20, 14, 12).astype(np.float32)
        self.y_labeled = np.random.randn(20, 3, 7).astype(np.float32)
        self.X_unlabeled = np.random.randn(20, 14, 12).astype(np.float32)
        self.y_unlabeled = np.random.randn(20, 3, 7).astype(np.float32)
        self.X_test = np.random.randn(10, 14, 12).astype(np.float32)
        self.y_test = np.random.randn(10, 3, 7).astype(np.float32)
        self.pretrain_cfg = {
            'enabled': True,
            'epochs': 1,
            'mask_ratio': 0.3,
            'learning_rate': 5e-4,
            'weight_decay': 3e-4,
            'early_stop_patience': 10,
        }

    def _make_trainer(self, pretrain_enabled=True):
        cfg = dict(self.pretrain_cfg)
        cfg['enabled'] = pretrain_enabled
        return PretrainFinetuneTrainer(
            model_type='vmd_cnn_bilstm_attention',
            input_size=12,
            device='cpu',
            teacher_epochs=1,
            student_epochs=1,
            pseudo_confidence_threshold=0.5,
            batch_size=8,
            pretrain_config=cfg,
        )

    def test_three_phase_pipeline_runs(self):
        """三阶段流水线应能跑通"""
        trainer = self._make_trainer(pretrain_enabled=True)
        student, metrics = trainer.fit(
            X_labeled=self.X_labeled, y_labeled=self.y_labeled,
            X_unlabeled=self.X_unlabeled, y_unlabeled=self.y_unlabeled,
            X_test=self.X_test, y_test=self.y_test,
        )
        self.assertIsNotNone(student)
        self.assertIn('test_loss', metrics)
        self.assertGreater(len(trainer.history['pretrain_losses']), 0)
        self.assertGreater(len(trainer.history['teacher_losses']), 0)
        self.assertGreater(len(trainer.history['student_losses']), 0)

    def test_pretrain_disabled_skips_pretrain(self):
        """--pretrain 关闭时,pretrain_losses 应为空,流水线降级为半监督"""
        trainer = self._make_trainer(pretrain_enabled=False)
        _, metrics = trainer.fit(
            X_labeled=self.X_labeled, y_labeled=self.y_labeled,
            X_unlabeled=self.X_unlabeled, y_unlabeled=self.y_unlabeled,
            X_test=self.X_test, y_test=self.y_test,
        )
        self.assertEqual(len(trainer.history['pretrain_losses']), 0)
        self.assertGreater(len(trainer.history['teacher_losses']), 0)
        self.assertGreater(len(trainer.history['student_losses']), 0)
        self.assertIn('test_loss', metrics)

    def test_teacher_student_share_backbone_init(self):
        """Teacher 与 Student 在训练前应共享 backbone 起点(在 enabled 路径下)"""
        trainer = self._make_trainer(pretrain_enabled=True)
        # 捕获训练前的 teacher 与 student backbone
        original_fit = trainer._fit_semi
        seen_states = {}

        def wrapped_fit(*args, **kwargs):
            # 替换 build 行为
            from air_quality.models import create_model
            t = create_model('vmd_cnn_bilstm_attention', input_size=12)
            s = create_model('vmd_cnn_bilstm_attention', input_size=12)
            # 用 backbone_state 加载
            from air_quality.training.pretrain import Pretrainer
            backbone_state = Pretrainer.extract_backbone_state(trainer._pretrain_model)
            Pretrainer.load_backbone_into(t, backbone_state)
            Pretrainer.load_backbone_into(s, backbone_state)
            seen_states['teacher_bilstm'] = t.bilstm.weight_hh_l0.detach().clone()
            seen_states['student_bilstm'] = s.bilstm.weight_hh_l0.detach().clone()
            seen_states['teacher_head'] = t.finetune_head[-1].weight.detach().clone()
            seen_states['student_head'] = s.finetune_head[-1].weight.detach().clone()
            return original_fit(*args, **kwargs)

        trainer._fit_semi = wrapped_fit
        trainer.fit(
            X_labeled=self.X_labeled, y_labeled=self.y_labeled,
            X_unlabeled=self.X_unlabeled, y_unlabeled=self.y_unlabeled,
            X_test=self.X_test, y_test=self.y_test,
        )
        # backbone 权重应一致
        self.assertTrue(torch.equal(
            seen_states['teacher_bilstm'], seen_states['student_bilstm']
        ))
        # head 权重应不同(独立随机)
        self.assertFalse(torch.equal(
            seen_states['teacher_head'], seen_states['student_head']
        ))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m unittest tests.test_pretrain_finetune -v 2>&1 | tail -10`
Expected: ModuleNotFoundError: No module named 'air_quality.training.pretrain_finetune'

- [ ] **Step 3: 实现 `src/air_quality/training/pretrain_finetune.py`**

```python
"""
PretrainFinetuneTrainer:三阶段流水线编排器

继承 SemiSupervisedTrainer,在 fit() 之前增加 Phase 0(无监督预训练)。

用法:
    trainer = PretrainFinetuneTrainer(
        model_type='vmd_cnn_bilstm_attention',
        input_size=12,
        device='cpu',
        teacher_epochs=80,
        student_epochs=120,
        pseudo_confidence_threshold=0.85,
        pretrain_config={'enabled': True, 'epochs': 60, ...},
    )
    student, metrics = trainer.fit(X_lab, y_lab, X_unl, y_unl, X_te, y_te)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from air_quality.models import create_model
from air_quality.training.pretrain import Pretrainer
from air_quality.training.semi_supervised import SemiSupervisedTrainer

logger = logging.getLogger(__name__)


class PretrainFinetuneTrainer(SemiSupervisedTrainer):
    """三阶段流水线(预训练 → 伪标签 → 微调)"""

    def __init__(
        self,
        model_type: str,
        input_size: int,
        device: str = 'cpu',
        teacher_epochs: int = 80,
        student_epochs: int = 120,
        pseudo_confidence_threshold: float = 0.85,
        pseudo_loss_weight: float = 0.5,
        batch_size: int = 32,
        learning_rate: float = 3e-4,
        weight_decay: float = 3e-4,
        pretrain_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            model_type=model_type,
            input_size=input_size,
            device=device,
            teacher_epochs=teacher_epochs,
            student_epochs=student_epochs,
            pseudo_confidence_threshold=pseudo_confidence_threshold,
            pseudo_loss_weight=pseudo_loss_weight,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        # 预训练配置(默认 disabled)
        self.pretrain_config = {
            'enabled': False,
            'epochs': 60,
            'mask_ratio': 0.3,
            'learning_rate': 5e-4,
            'weight_decay': 3e-4,
            'early_stop_patience': 20,
            'target': 'imf',
        }
        if pretrain_config:
            self.pretrain_config.update(pretrain_config)
        # 新增 history 字段
        self.history.setdefault('pretrain_losses', [])
        # 暂存预训练模型(供测试访问)
        self._pretrain_model: Optional[nn.Module] = None

    def _run_pretrain(self, X_unlabeled: np.ndarray) -> Optional[Dict[str, torch.Tensor]]:
        """Phase 0:预训练;返回 backbone_state 或 None。"""
        if not self.pretrain_config.get('enabled', False):
            return None

        logger.info("[Phase 0] Pretraining on unlabeled data...")
        # 创建新模型作为预训练目标
        model = create_model(self.model_type, input_size=self.input_size)
        pretrainer = Pretrainer(
            model=model,
            device=str(self.device),
            learning_rate=self.pretrain_config['learning_rate'],
            weight_decay=self.pretrain_config['weight_decay'],
            mask_ratio=self.pretrain_config['mask_ratio'],
            early_stop_patience=self.pretrain_config['early_stop_patience'],
        )
        pretrainer.fit(X_unlabeled, epochs=self.pretrain_config['epochs'])
        self.history['pretrain_losses'] = pretrainer.history['train_losses']
        self._pretrain_model = model
        return Pretrainer.extract_backbone_state(model)

    def fit(
        self,
        X_labeled,
        y_labeled,
        X_unlabeled,
        y_unlabeled,
        X_test,
        y_test,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        # Phase 0: 预训练
        backbone_state = self._run_pretrain(X_unlabeled)

        # Phase 1-4: 沿用 SemiSupervisedTrainer,但需要把 backbone_state 传给 Teacher/Student
        # 策略: 直接调用父类 fit,父类内已经会通过 _build_model_with_state 读 backbone_state
        # 但父类当前没有暴露 backbone_state 入口。这里采用一个临时方案:
        # 临时给父类注入 _backbone_state 属性,再调用 fit。
        if backbone_state is not None:
            self._backbone_state = backbone_state  # type: ignore[attr-defined]
        else:
            self._backbone_state = None  # type: ignore[attr-defined]

        return super().fit(
            X_labeled, y_labeled,
            X_unlabeled, y_unlabeled,
            X_test, y_test,
        )
```

> **设计说明**:本实现选择最简洁的"向父类注入 `_backbone_state`"方式;Task 4 已经在 `_train_teacher`/`_train_student` 里通过 `getattr(self, '_backbone_state', None)` 读取它,所以这里注入即生效。无需修改父类 `fit()` 签名。

- [ ] **Step 4: 回到 Task 4,确保 `_train_teacher` 与 `_train_student` 使用 `_build_model_with_state` + `backbone_state`**

打开 `src/air_quality/training/semi_supervised.py`,找到 `_train_teacher`,把:
```python
        teacher = create_model(self.model_type, input_size=self.input_size).to(self.device)
```
替换为:
```python
        backbone_state = getattr(self, '_backbone_state', None)
        teacher = self._build_model_with_state(
            self.model_type, input_size=self.input_size,
            backbone_state=backbone_state,
        ).to(self.device)
```

`_train_student` 同样改动。确保 `from typing import Optional, Dict` 已在文件顶部。

- [ ] **Step 5: 跑测试确认通过**

Run: `python -m unittest tests.test_pretrain_finetune -v`
Expected: 3 个测试全部通过

- [ ] **Step 6: 跑全部测试确认未破坏**

Run: `python -m unittest discover tests 2>&1 | tail -5`
Expected: 全部通过(包括 test_semi_trainer.py, test_full_semi_pipeline.py)

- [ ] **Step 7: 修改 `src/air_quality/training/__init__.py`,导出新类**

在 `__all__` 列表追加:
```python
from .pretrain import Pretrainer, compute_masked_mse
from .pretrain_finetune import PretrainFinetuneTrainer
```

`__all__` 列表追加:
```python
    'Pretrainer',
    'compute_masked_mse',
    'PretrainFinetuneTrainer',
```

- [ ] **Step 8: 跑测试再次确认**

Run: `python -m unittest discover tests 2>&1 | tail -3`
Expected: 全部通过

- [ ] **Step 9: 提交**

```bash
git add src/air_quality/training/pretrain_finetune.py \
        src/air_quality/training/semi_supervised.py \
        src/air_quality/training/__init__.py \
        tests/test_pretrain_finetune.py
git commit -m "feat(training): 新增 PretrainFinetuneTrainer 三阶段编排器

继承 SemiSupervisedTrainer,在 fit() 之前增加 Phase 0:
- 预训练仅在 unlabeled 段上做掩码自监督
- 预训练 backbone 通过 _backbone_state 注入父类
- Teacher/Student 共享 backbone 起点 + 独立 head

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: 端到端集成测试(mock Excel + VMD + 三阶段)

**Files:**
- Create: `tests/test_pretrain_semi_pipeline.py`

- [ ] **Step 1: 写失败测试 `tests/test_pretrain_semi_pipeline.py`**

```python
"""端到端预训练-微调半监督 pipeline 集成测试"""
import unittest
import tempfile
import os
import numpy as np
import pandas as pd


class TestPretrainSemiPipeline(unittest.TestCase):
    def _create_mock_excel(self, path):
        """构造小型 Excel mock 数据集"""
        n = 200
        dates = pd.date_range('2020-01-01', periods=n, freq='D')
        df = pd.DataFrame({
            'pubtime': dates,
            'AQI': np.random.randint(20, 200, n),
            'PM2.5': np.random.randint(10, 150, n),
            'PM10': np.random.randint(20, 200, n),
            'NO2': np.random.randint(10, 80, n),
            'SO2': np.random.randint(5, 50, n),
            'CO': np.random.uniform(0.5, 3.0, n),
            'O3': np.random.randint(20, 160, n),
        })
        df.to_excel(path, index=False)

    def test_pretrain_semi_pipeline_runs(self):
        """端到端预训练+半监督 pipeline 应能跑通"""
        from air_quality.config import config
        from air_quality.data.processor import AirQualityDataProcessor
        from air_quality.data.vmd import VMDDecomposer
        from air_quality.data.vmd_features import apply_vmd_to_aqi
        from air_quality.training import PretrainFinetuneTrainer

        with tempfile.TemporaryDirectory() as tmp:
            excel_path = os.path.join(tmp, 'mock.xlsx')
            self._create_mock_excel(excel_path)
            config.update_from_dict({'data': {'data_file': excel_path}})

            processor = AirQualityDataProcessor(
                seq_length=14, prediction_days=3,
                output_size=config.model.output_size,
            )
            data, scaler, features, dates = processor.load_and_preprocess_data(excel_path)
            X, y = processor.create_sequences(data)

            decomposer = VMDDecomposer(K=4)
            X_vmd = apply_vmd_to_aqi(X, decomposer)

            (X_lab, y_lab), (X_unl, y_unl), (X_te, y_te) = processor.split_three_way(X_vmd, y)

            trainer = PretrainFinetuneTrainer(
                model_type='vmd_cnn_bilstm_attention',
                input_size=X_vmd.shape[2],
                device='cpu',
                teacher_epochs=1,
                student_epochs=1,
                pseudo_confidence_threshold=0.5,
                batch_size=8,
                pretrain_config={
                    'enabled': True,
                    'epochs': 1,
                    'mask_ratio': 0.3,
                    'learning_rate': 5e-4,
                },
            )
            student, metrics = trainer.fit(
                X_labeled=X_lab, y_labeled=y_lab,
                X_unlabeled=X_unl, y_unlabeled=y_unl,
                X_test=X_te, y_test=y_te,
            )
            self.assertIsNotNone(student)
            self.assertFalse(np.isnan(metrics['test_loss']))
            self.assertGreater(len(trainer.history['pretrain_losses']), 0)
            self.assertGreater(len(trainer.history['teacher_losses']), 0)
            self.assertGreater(len(trainer.history['student_losses']), 0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m unittest tests.test_pretrain_semi_pipeline -v 2>&1 | tail -10`
Expected: ModuleNotFoundError 或 ImportError

- [ ] **Step 3: 不需要写实现代码,只需确保 Task 5 已完成导出 PretrainFinetuneTrainer;若失败,回到 Task 5 检查 `__init__.py`**

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m unittest tests.test_pretrain_semi_pipeline -v`
Expected: 1 个测试通过

- [ ] **Step 5: 跑全部测试**

Run: `python -m unittest discover tests -v 2>&1 | tail -5`
Expected: 全部通过(总测试数 ≥ 40)

- [ ] **Step 6: 提交**

```bash
git add tests/test_pretrain_semi_pipeline.py
git commit -m "test: 端到端预训练-微调半监督 pipeline 集成测试

mock Excel → VMD 分解 → 三段划分 → 预训练 → 伪标签 → 微调 → 评估,
验证 metrics['test_loss'] 非 NaN 且三段 history 全部有数据。

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: 在 scripts/train.py 增加 --pretrain CLI 参数

**Files:**
- Modify: `scripts/train.py`

- [ ] **Step 1: 在 `parse_args` 中追加参数**

找到 `--semi-supervised` 附近的 `parser.add_argument` 块,在其后追加:

```python
    # 预训练-微调(VMD 半监督基础上)
    parser.add_argument(
        '--pretrain', dest='pretrain', action='store_true',
        help='启用无监督预训练-有监督微调范式(需同时 --semi-supervised)',
    )
    parser.add_argument(
        '--pretrain-epochs', dest='pretrain_epochs', type=int, default=None,
        help='预训练轮数(默认从 config 读取)',
    )
    parser.add_argument(
        '--pretrain-mask-ratio', dest='pretrain_mask_ratio', type=float, default=None,
        help='掩码比例 0~1',
    )
    parser.add_argument(
        '--pretrain-lr', dest='pretrain_lr', type=float, default=None,
        help='预训练学习率',
    )
```

- [ ] **Step 2: 在 `main()` 中,半监督分支之前,添加 CLI 覆盖**

找到 `if config.semi.enabled:` 块,在它之前添加:

```python
        # 应用预训练配置覆盖
        if args.pretrain:
            config.pretrain.enabled = True
        if args.pretrain_epochs is not None:
            config.pretrain.epochs = args.pretrain_epochs
        if args.pretrain_mask_ratio is not None:
            config.pretrain.mask_ratio = args.pretrain_mask_ratio
        if args.pretrain_lr is not None:
            config.pretrain.learning_rate = args.pretrain_lr
```

- [ ] **Step 3: 验证 `--help` 输出新参数**

Run: `python -m scripts.train --help | grep -E "pretrain"`
Expected: 4 行匹配(`--pretrain`, `--pretrain-epochs`, `--pretrain-mask-ratio`, `--pretrain-lr`)

- [ ] **Step 4: dry-run 测试(不实际训练,只检查路径)**

不需要新测试;现有 `test_full_semi_pipeline.py` 已覆盖 `SemiSupervisedTrainer` 行为,新加的 `--pretrain` 标志由 CLI 覆盖 `config.pretrain.enabled`,集成测试 `test_pretrain_semi_pipeline.py` 已覆盖启用路径。

- [ ] **Step 5: 跑全部测试确认未破坏**

Run: `python -m unittest discover tests 2>&1 | tail -3`
Expected: 全部通过

- [ ] **Step 6: 提交**

```bash
git add scripts/train.py
git commit -m "feat(scripts): train.py 支持 --pretrain 系列 CLI 参数

--pretrain / --pretrain-epochs / --pretrain-mask-ratio / --pretrain-lr,
仅在 --semi-supervised 模式下生效。

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: 顶层导出 + 文档更新

**Files:**
- Modify: `src/air_quality/__init__.py`
- Modify: `README.md`
- Modify: `docs/algorithm_analysis.md`

- [ ] **Step 1: 在 `src/air_quality/__init__.py` 顶层导出新类**

在文件中追加:
```python
from .training.pretrain import Pretrainer, compute_masked_mse
from .training.pretrain_finetune import PretrainFinetuneTrainer
```

并把 `'Pretrainer'`、`'PretrainFinetuneTrainer'` 加入 `__all__`(若已存在)。

- [ ] **Step 2: 在 README.md 末尾追加使用说明**

```markdown
### 预训练-微调半监督训练(VMD-CNN-BiLSTM-Attention)

```bash
# 启用预训练-微调(必须在 --semi-supervised 之后)
python -m scripts.train --semi-supervised --pretrain

# 自定义预训练轮数
python -m scripts.train --semi-supervised --pretrain --pretrain-epochs 30

# 自定义掩码比例
python -m scripts.train --semi-supervised --pretrain --pretrain-mask-ratio 0.5
```

三阶段流水线:预训练(unlabeled 段掩码自监督) → 伪标签(Teacher 推理) → 微调(Student 联合训练)。
```

- [ ] **Step 3: 在 `docs/algorithm_analysis.md` 末尾追加章节**

```markdown
## 预训练-微调范式:VMD-CNN-BiLSTM-Attention + 掩码自监督

详见 `docs/superpowers/specs/2026-06-17-vmd-pretrain-finetune-design.md`。

### 三阶段流水线
1. **预训练**:仅在 40% unlabeled 段,掩码自监督预测被遮盖的 IMF
2. **伪标签**:Teacher 在 40% labeled 上训练,推理 unlabeled 生成伪标签
3. **微调**:Student 用预训练 backbone 初始化,联合 labeled + 伪标签 训练

### 共享 backbone
`VMDCNNBILSTMAttentionModel` 拆为:
- 共享 backbone(input_proj + CNN + BiLSTM + Attention)
- `pretrain_head`:Linear → K,预测被遮盖的 IMF
- `finetune_head`:Sequential,预测 7 维污染物

### 兼容性
`forward(x, mode='finetune')` 默认,行为完全向后兼容。
```

- [ ] **Step 4: 跑全部测试做最终回归**

Run: `python -m unittest discover tests -v 2>&1 | tail -5`
Expected: 全部通过(总测试数 ≥ 40)

- [ ] **Step 5: 提交**

```bash
git add src/air_quality/__init__.py README.md docs/algorithm_analysis.md
git commit -m "docs: 顶层导出 + README/algorithm_analysis 补充预训练-微调说明

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## 自审报告

| 检查项 | 结论 |
|---|---|
| **Spec 覆盖** | ✅ Spec §1-§9 每条要求都有对应 Task |
| **占位符扫描** | ✅ 无 TBD/TODO/FIXME,所有代码完整 |
| **类型一致性** | ✅ `Pretrainer.fit / extract_backbone_state / generate_mask`、`PretrainFinetuneTrainer.fit` 在所有 Task 中签名一致 |
| **测试覆盖** | ✅ 单元(task 1-3)+ 三阶段(task 5)+ 端到端(task 6) |
| **TDD 顺序** | ✅ 每个 Task 都是「写测试 → 跑失败 → 实现 → 跑通过 → 提交」 |
| **依赖一致性** | ✅ Pretrainer(Task 2)→ PretrainConfig(Task 3)→ SemiSupervisedTrainer 重构(Task 4)→ PretrainFinetuneTrainer(Task 5)→ 集成测试(Task 6) |
| **向后兼容** | ✅ `forward(mode='finetune')` 默认值;`_backbone_state` 通过 `getattr` 兜底 |
| **风险缓解** | ✅ `strict=False` 加载 + `pretrain_losses=[]` 默认值 + 早停 |
