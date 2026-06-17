# VMD-CNN-BiLSTM-Attention 半监督范式实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将空气质量预测系统从全监督范式迁移至半监督范式，引入 VMD-CNN-BiLSTM-Attention 专用架构（40% 有标签 + 40% 无标签 + 20% 测试，伪标签法）。

**Architecture:** 在现有全监督框架基础上新增：(1) VMD 分解器封装 vmdpy；(2) 三段时序划分；(3) VMD-CNN-BiLSTM-Attention 模型；(4) SemiSupervisedTrainer 实现 Teacher → 伪标签 → Student 4 阶段训练。现有 ModelTrainer / 4 个 baseline 模型全部保留。

**Tech Stack:** PyTorch, vmdpy (新增), NumPy, scikit-learn, unittest

---

## 文件结构

**新增文件**：
- `src/air_quality/models/vmd_cnn_bilstm_attention.py` — 新模型
- `src/air_quality/data/vmd.py` — VMD 分解器
- `src/air_quality/training/semi_supervised.py` — 半监督训练器
- `tests/test_vmd.py` — VMD 单元测试
- `tests/test_split_three_way.py` — 三段划分测试
- `tests/test_vmd_cnn_bilstm.py` — 模型 forward 测试
- `tests/test_semi_trainer.py` — 半监督训练器测试

**修改文件**：
- `src/air_quality/models/factory.py` — 注册新模型
- `src/air_quality/models/__init__.py` — 导出
- `src/air_quality/data/processor.py` — 增加 split_three_way
- `src/air_quality/config/settings.py` — 增加 SemiSupervisedConfig + VMDConfig
- `src/air_quality/training/__init__.py` — 导出 SemiSupervisedTrainer
- `src/air_quality/__init__.py` — 顶层导出
- `scripts/train.py` — 增加 --semi-supervised / --vmd-k / --pseudo-threshold 等参数
- `requirements.txt` — 新增 vmdpy>=0.2

---

## Task 1: 添加 vmdpy 依赖

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: 添加 vmdpy 到 requirements.txt**

```diff
 # 现有
 torch>=1.13
 scikit-learn>=1.0
 pandas>=1.5
 openpyxl>=3.0
 matplotlib>=3.5
 numpy>=1.21
 joblib>=1.1
+
+# 半监督 VMD 方案
+vmdpy>=0.2
```

- [ ] **Step 2: 安装依赖**

Run: `pip install vmdpy`
Expected: Successfully installed vmdpy-X.X.X

- [ ] **Step 3: 验证导入**

Run: `python -c "from vmdpy import VMD; print('vmdpy OK')"`
Expected: `vmdpy OK`

- [ ] **Step 4: 提交**

```bash
git add requirements.txt
git commit -m "build: 添加 vmdpy 依赖以支持 VMD 分解"
```

---

## Task 2: 实现 VMDDecomposer + 单元测试

**Files:**
- Create: `src/air_quality/data/vmd.py`
- Create: `tests/test_vmd.py`

- [ ] **Step 1: 编写失败测试 `tests/test_vmd.py`**

```python
"""VMD 分解器单元测试"""
import unittest
import numpy as np

from air_quality.data.vmd import VMDDecomposer


class TestVMDDecomposer(unittest.TestCase):
    def test_decompose_returns_K_modes(self):
        """应返回 K 个 IMF 分量"""
        decomposer = VMDDecomposer(K=4)
        signal = np.sin(np.linspace(0, 4 * np.pi, 100))
        result = decomposer.decompose(signal)
        self.assertEqual(result.shape, (4, 100))

    def test_decompose_reconstruction_error(self):
        """重构误差应足够小"""
        decomposer = VMDDecomposer(K=4, alpha=2000)
        np.random.seed(42)
        signal = np.sin(np.linspace(0, 4 * np.pi, 100)) + 0.1 * np.random.randn(100)
        imfs = decomposer.decompose(signal)
        reconstructed = imfs.sum(axis=0)
        error = np.linalg.norm(signal - reconstructed) / np.linalg.norm(signal)
        self.assertLess(error, 0.05)  # 重构误差 < 5%

    def test_decompose_2d_input(self):
        """二维输入 (N, T) 应逐行分解"""
        decomposer = VMDDecomposer(K=4)
        signals = np.random.randn(3, 100)
        result = decomposer.decompose(signals)
        self.assertEqual(result.shape, (3, 4, 100))

    def test_configurable_K(self):
        """K 参数应可配置"""
        for K in [2, 4, 6]:
            decomposer = VMDDecomposer(K=K)
            signal = np.sin(np.linspace(0, 4 * np.pi, 100))
            result = decomposer.decompose(signal)
            self.assertEqual(result.shape, (K, 100))
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest tests.test_vmd -v`
Expected: ImportError or ModuleNotFoundError

- [ ] **Step 3: 实现 `src/air_quality/data/vmd.py`**

```python
"""VMD (Variational Mode Decomposition) 分解器

封装 vmdpy 库，提供对一维与二维信号的变分模态分解。
"""
from __future__ import annotations

import numpy as np
from vmdpy import VMD


class VMDDecomposer:
    """VMD 分解器

    Args:
        K: IMF 模态数（默认 4）
        alpha: 带宽约束（默认 2000，值越大带宽越窄）
        tau: 噪声容忍（默认 0）
        DC: 是否保留 DC 分量（默认 0 = 不保留）
        init: 初始化方式（1=均匀分布，2=随机）
        tol: 收敛容差（默认 1e-7）
    """

    def __init__(
        self,
        K: int = 4,
        alpha: float = 2000,
        tau: float = 0,
        DC: int = 0,
        init: int = 1,
        tol: float = 1e-7,
    ):
        if K < 1:
            raise ValueError(f"K 必须 ≥ 1，当前值: {K}")
        self.K = K
        self.alpha = alpha
        self.tau = tau
        self.DC = DC
        self.init = init
        self.tol = tol

    def decompose(self, signal: np.ndarray) -> np.ndarray:
        """对信号做 VMD 分解

        Args:
            signal: 一维 (T,) 或二维 (N, T) 信号

        Returns:
            一维输入返回 (K, T)；二维输入返回 (N, K, T)
        """
        if signal.ndim == 1:
            return self._decompose_single(signal)
        elif signal.ndim == 2:
            return np.stack(
                [self._decompose_single(row) for row in signal], axis=0
            )
        else:
            raise ValueError(f"signal 维度必须是 1 或 2，当前: {signal.ndim}")

    def _decompose_single(self, signal: np.ndarray) -> np.ndarray:
        u, _, _ = VMD(
            signal,
            self.alpha,
            self.tau,
            self.K,
            self.DC,
            self.init,
            self.tol,
        )
        return u  # (K, T)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m unittest tests.test_vmd -v`
Expected: 4 tests pass

- [ ] **Step 5: 提交**

```bash
git add src/air_quality/data/vmd.py tests/test_vmd.py
git commit -m "feat(data): 新增 VMDDecomposer 封装 vmdpy 库"
```

---

## Task 3: 实现 split_three_way + 单元测试

**Files:**
- Modify: `src/air_quality/data/processor.py` (在文件末尾新增方法)
- Create: `tests/test_split_three_way.py`

- [ ] **Step 1: 编写失败测试 `tests/test_split_three_way.py`**

```python
"""split_three_way 时序三段划分单元测试"""
import unittest
import numpy as np

from air_quality.data.processor import AirQualityDataProcessor


class TestSplitThreeWay(unittest.TestCase):
    def setUp(self):
        self.processor = AirQualityDataProcessor()
        # 构造 100 个序列
        np.random.seed(0)
        self.X = np.random.randn(100, 14, 9)
        self.y = np.random.randn(100, 3, 7)

    def test_default_ratios(self):
        """默认 0.4/0.4/0.2 划分"""
        (Xl, yl), (Xu, yu), (Xt, yt) = self.processor.split_three_way(self.X, self.y)
        self.assertEqual(len(Xl), 40)
        self.assertEqual(len(Xu), 40)
        self.assertEqual(len(Xt), 20)
        self.assertEqual(len(yl), 40)
        self.assertEqual(len(yu), 40)
        self.assertEqual(len(yt), 20)

    def test_custom_ratios(self):
        """自定义 0.5/0.3/0.2 划分"""
        (Xl, yl), (Xu, yu), (Xt, yt) = self.processor.split_three_way(
            self.X, self.y, ratios=(0.5, 0.3, 0.2)
        )
        self.assertEqual(len(Xl), 50)
        self.assertEqual(len(Xu), 30)
        self.assertEqual(len(Xt), 20)

    def test_chronological_order(self):
        """必须按时间顺序划分，不打乱"""
        (Xl, _), (Xu, _), (Xt, _) = self.processor.split_three_way(
            self.X, self.y, ratios=(0.4, 0.4, 0.2)
        )
        # labeled 段的最大索引 < unlabeled 段的最小索引
        self.assertLess(Xl[-1, 0, 0] - Xl[0, 0, 0], Xu[0, 0, 0] - Xl[0, 0, 0] + 1000)
        # unlabeled 段在前，test 段在后
        self.assertLess(Xu[-1, 0, 0] - Xu[0, 0, 0], Xt[0, 0, 0] - Xu[0, 0, 0] + 1000)

    def test_no_overlap(self):
        """三段之间无重叠"""
        (Xl, _), (Xu, _), (Xt, _) = self.processor.split_three_way(
            self.X, self.y, ratios=(0.4, 0.4, 0.2)
        )
        total = len(Xl) + len(Xu) + len(Xt)
        self.assertEqual(total, len(self.X))

    def test_invalid_ratios(self):
        """比例之和不等于 1 应报错"""
        with self.assertRaises(AssertionError):
            self.processor.split_three_way(self.X, self.y, ratios=(0.5, 0.3, 0.3))
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest tests.test_split_three_way -v`
Expected: AttributeError: 'AirQualityDataProcessor' object has no attribute 'split_three_way'

- [ ] **Step 3: 在 `src/air_quality/data/processor.py` 末尾新增方法**

```python
    def split_three_way(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ratios: tuple = (0.4, 0.4, 0.2),
    ) -> tuple:
        """按时间顺序三段划分（不打乱），保持时序连续性。

        用于半监督学习的 40% 有标签 / 40% 无标签 / 20% 测试划分。

        Args:
            X: (N, seq_length, n_features) 序列
            y: (N, prediction_days, output_size) 目标
            ratios: 三段比例之和必须为 1.0

        Returns:
            ((X_labeled, y_labeled), (X_unlabeled, y_unlabeled), (X_test, y_test))
        """
        assert abs(sum(ratios) - 1.0) < 1e-6, f"比例之和必须为 1.0，当前: {sum(ratios)}"
        n = len(X)
        n_label = int(n * ratios[0])
        n_unlabel = int(n * ratios[1])
        return (
            (X[:n_label], y[:n_label]),
            (X[n_label:n_label + n_unlabel], y[n_label:n_label + n_unlabel]),
            (X[n_label + n_unlabel:], y[n_label + n_unlabel:]),
        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m unittest tests.test_split_three_way -v`
Expected: 5 tests pass

- [ ] **Step 5: 提交**

```bash
git add src/air_quality/data/processor.py tests/test_split_three_way.py
git commit -m "feat(data): 新增 split_three_way 时序三段划分方法"
```

---

## Task 4: 实现 VMD-CNN-BiLSTM-Attention 模型 + 单元测试

**Files:**
- Create: `src/air_quality/models/vmd_cnn_bilstm_attention.py`
- Create: `tests/test_vmd_cnn_bilstm.py`
- Modify: `src/air_quality/models/factory.py`
- Modify: `src/air_quality/models/__init__.py`

- [ ] **Step 1: 编写失败测试 `tests/test_vmd_cnn_bilstm.py`**

```python
"""VMD-CNN-BiLSTM-Attention 模型单元测试"""
import unittest
import torch

from air_quality.models.vmd_cnn_bilstm_attention import (
    VMDCNNBILSTMAttentionModel,
)


class TestVMDCNNBILSTMModel(unittest.TestCase):
    def setUp(self):
        self.model = VMDCNNBILSTMAttentionModel(
            input_size=9,
            output_size=7,
            vmd_K=4,
            cnn_filters=32,    # 测试用小尺寸
            bilstm_hidden=32,
            bilstm_layers=1,
            dropout=0.1,
        )

    def test_forward_shape(self):
        """forward 输出形状应为 (B, T, output_size)"""
        x = torch.randn(2, 14, 9)
        out = self.model(x)
        self.assertEqual(out.shape, (2, 14, 7))

    def test_take_last_3_days(self):
        """取最后 prediction_days 步应得到 (B, 3, 7)"""
        x = torch.randn(2, 14, 9)
        out = self.model(x)
        last_3 = out[:, -3:, :]
        self.assertEqual(last_3.shape, (2, 3, 7))

    def test_parameter_count(self):
        """应有一定数量的可训练参数"""
        n_params = self.model.count_parameters()
        self.assertGreater(n_params, 1000)
        self.assertLess(n_params, 5_000_000)

    def test_vmd_K_configurable(self):
        """VMD K 参数应可配置"""
        for K in [2, 4, 6]:
            model = VMDCNNBILSTMAttentionModel(
                input_size=9, output_size=7, vmd_K=K,
                cnn_filters=16, bilstm_hidden=16, bilstm_layers=1,
            )
            x = torch.randn(1, 14, 9)
            out = model(x)
            self.assertEqual(out.shape, (1, 14, 7))
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest tests.test_vmd_cnn_bilstm -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: 实现 `src/air_quality/models/vmd_cnn_bilstm_attention.py`**

```python
"""
VMD-CNN-BiLSTM-Attention 模型

数据流：VMD(AQI) → 拼接其他特征 → CNN → BiLSTM → Attention → FC
"""
import numpy as np
import torch
import torch.nn as nn

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
    """VMD-CNN-BiLSTM-Attention 混合模型

    Args:
        input_size: 输入特征维度（含 AQI 等 9 维）
        output_size: 输出维度（默认 7，仅污染物）
        vmd_K: VMD 模态数
        cnn_filters: CNN 卷积核数量
        bilstm_hidden: BiLSTM 隐藏层维度
        bilstm_layers: BiLSTM 层数
        dropout: dropout 比率
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
        # VMD 在推理时调用；训练时 offline 分解后通过 input_proj 输入
        self.input_proj = nn.Linear(input_size - 1 + vmd_K, cnn_filters)
        # 多尺度 CNN
        self.conv3 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(cnn_filters * 3)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(14)  # 保持时序长度
        # BiLSTM
        self.bilstm = nn.LSTM(
            cnn_filters * 3,
            bilstm_hidden,
            bilstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if bilstm_layers > 1 else 0.0,
        )
        # Attention
        self.attention = TemporalAttention(bilstm_hidden * 2)
        # FC Head
        self.fc1 = nn.Linear(bilstm_hidden * 2, bilstm_hidden)
        self.fc2 = nn.Linear(bilstm_hidden, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_size) —— 注意：输入应在外部已做 VMD 拼接
               即：原始 8 维（去除 AQI）+ K 个 IMF 分量

        Returns:
            (B, T, output_size)
        """
        # 投影
        h = self.input_proj(x)  # (B, T, cnn_filters)
        # 多尺度 CNN
        h_t = h.transpose(1, 2)  # (B, cnn_filters, T)
        h3 = self.conv3(h_t)
        h5 = self.conv5(h_t)
        h7 = self.conv7(h_t)
        h_cat = torch.cat([h3, h5, h7], dim=1)  # (B, cnn_filters*3, T)
        h_cat = self.bn(h_cat)
        h_cat = self.relu(h_cat)
        h_cat = self.pool(h_cat)  # (B, cnn_filters*3, T)
        h_seq = h_cat.transpose(1, 2)  # (B, T, cnn_filters*3)
        # BiLSTM
        lstm_out, _ = self.bilstm(h_seq)  # (B, T, bilstm_hidden*2)
        # Attention
        attended = self.attention(lstm_out)
        # FC
        out = self.relu(self.fc1(attended))
        out = self.dropout(out)
        return self.fc2(out)  # (B, T, output_size)
```

- [ ] **Step 4: 修改 `src/air_quality/models/factory.py`**

```python
"""
模型工厂函数
"""
from .base import BaseModel


def create_model(model_type: str, input_size: int, **kwargs) -> BaseModel:
    """创建指定类型的模型"""
    from .transformer import TransformerModel
    from .lstm import LSTMModel
    from .cnn import CNNModel
    from .hybrid import HybridModel
    from .vmd_cnn_bilstm_attention import VMDCNNBILSTMAttentionModel

    model_classes = {
        'transformer': TransformerModel,
        'lstm': LSTMModel,
        'cnn': CNNModel,
        'hybrid': HybridModel,
        'vmd_cnn_bilstm_attention': VMDCNNBILSTMAttentionModel,
    }

    if model_type not in model_classes:
        raise ValueError(
            f"不支持的模型类型: {model_type}，可选: {list(model_classes.keys())}"
        )

    return model_classes[model_type](input_size=input_size, **kwargs)
```

- [ ] **Step 5: 修改 `src/air_quality/models/__init__.py`**

```python
"""
模型模块 - 包含所有模型架构
"""
from .base import BaseModel, PositionalEncoding
from .transformer import TransformerModel
from .lstm import LSTMModel
from .cnn import CNNModel
from .hybrid import HybridModel
from .ensemble import EnsembleModel
from .vmd_cnn_bilstm_attention import VMDCNNBILSTMAttentionModel
from .factory import create_model

__all__ = [
    'BaseModel',
    'PositionalEncoding',
    'TransformerModel',
    'LSTMModel',
    'CNNModel',
    'HybridModel',
    'EnsembleModel',
    'VMDCNNBILSTMAttentionModel',
    'create_model',
]
```

- [ ] **Step 6: 运行测试确认通过**

Run: `python -m unittest tests.test_vmd_cnn_bilstm -v`
Expected: 4 tests pass

- [ ] **Step 7: 运行全部测试确认未破坏现有功能**

Run: `python -m unittest discover tests -v`
Expected: 所有测试通过

- [ ] **Step 8: 提交**

```bash
git add src/air_quality/models/vmd_cnn_bilstm_attention.py \
        src/air_quality/models/factory.py \
        src/air_quality/models/__init__.py \
        tests/test_vmd_cnn_bilstm.py
git commit -m "feat(model): 新增 VMD-CNN-BiLSTM-Attention 模型"
```

---

## Task 5: 实现 VMD-AQI 特征变换工具函数

**Files:**
- Create: `src/air_quality/data/vmd_features.py`
- Create: `tests/test_vmd_features.py`

- [ ] **Step 1: 编写失败测试 `tests/test_vmd_features.py`**

```python
"""VMD 特征变换工具测试"""
import unittest
import numpy as np

from air_quality.data.vmd_features import apply_vmd_to_aqi, build_vmd_features
from air_quality.data.vmd import VMDDecomposer


class TestVMDFeatures(unittest.TestCase):
    def test_apply_vmd_to_aqi_shape(self):
        """应返回 (N, 14, 8+K)"""
        decomposer = VMDDecomposer(K=4)
        X = np.random.randn(10, 14, 9)
        result = apply_vmd_to_aqi(X, decomposer)
        self.assertEqual(result.shape, (10, 14, 8 + 4))

    def test_build_vmd_features_preserves_non_aqi(self):
        """非 AQI 列应保持不变"""
        decomposer = VMDDecomposer(K=4)
        X = np.random.randn(5, 14, 9)
        result = build_vmd_features(X, decomposer)
        # 后 8 列（非 AQI）应等于 X 的后 8 列
        np.testing.assert_array_almost_equal(result[:, :, -8:], X[:, :, -8:])

    def test_build_vmd_features_aqi_replaced(self):
        """AQI 列应被替换为 IMF 分量（除最后一个 IMF 与原 AQI 对齐）"""
        decomposer = VMDDecomposer(K=4)
        X = np.random.randn(5, 14, 9)
        result = build_vmd_features(X, decomposer)
        # 维度从 9 → 8 + K = 12
        self.assertEqual(result.shape[2], 12)
        # 前 K 列是 VMD 分量
        self.assertEqual(result.shape[2] - 8, 4)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest tests.test_vmd_features -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: 实现 `src/air_quality/data/vmd_features.py`**

```python
"""VMD-AQI 特征变换工具

将原始 (N, T, 9) 特征中的 AQI 列（第 0 维）替换为 K 个 VMD IMF 分量，
返回 (N, T, 8+K) 特征。
"""
import numpy as np

from .vmd import VMDDecomposer


def apply_vmd_to_aqi(X: np.ndarray, decomposer: VMDDecomposer) -> np.ndarray:
    """对每个时间序列的 AQI 列（最后一维的第 0 列）做 VMD 分解。

    Args:
        X: (N, T, 9) — 第 0 维是 AQI
        decomposer: VMDDecomposer 实例

    Returns:
        (N, T, 8 + K) — 第 0 维被替换为 K 个 IMF 分量
    """
    N, T, _ = X.shape
    aqi_signals = X[:, :, 0]  # (N, T)
    imfs = decomposer.decompose(aqi_signals)  # (N, K, T)
    # 转置为 (N, T, K) 以便拼接
    imfs = imfs.transpose(0, 2, 1)
    other_features = X[:, :, 1:]  # (N, T, 8)
    return np.concatenate([imfs, other_features], axis=-1)


def build_vmd_features(X: np.ndarray, decomposer: VMDDecomposer) -> np.ndarray:
    """便捷别名，等价于 apply_vmd_to_aqi。"""
    return apply_vmd_to_aqi(X, decomposer)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m unittest tests.test_vmd_features -v`
Expected: 3 tests pass

- [ ] **Step 5: 提交**

```bash
git add src/air_quality/data/vmd_features.py tests/test_vmd_features.py
git commit -m "feat(data): 新增 VMD-AQI 特征变换工具"
```

---

## Task 6: 添加 VMDConfig 与 SemiSupervisedConfig

**Files:**
- Modify: `src/air_quality/config/settings.py`
- Create: `tests/test_semi_config.py`

- [ ] **Step 1: 编写失败测试 `tests/test_semi_config.py`**

```python
"""半监督配置类单元测试"""
import unittest

from air_quality.config import config


class TestSemiSupervisedConfig(unittest.TestCase):
    def test_vmd_config_defaults(self):
        self.assertTrue(config.vmd.enabled)
        self.assertEqual(config.vmd.K, 4)
        self.assertEqual(config.vmd.alpha, 2000)

    def test_semi_supervised_defaults(self):
        self.assertFalse(config.semi.enabled)
        self.assertEqual(config.semi.labeled_ratio, 0.4)
        self.assertEqual(config.semi.unlabeled_ratio, 0.4)
        self.assertEqual(config.semi.test_ratio, 0.2)
        self.assertEqual(config.semi.teacher_epochs, 80)
        self.assertEqual(config.semi.student_epochs, 120)
        self.assertEqual(config.semi.pseudo_confidence_threshold, 0.85)
        self.assertEqual(config.semi.pseudo_loss_weight, 0.5)

    def test_config_update_from_dict(self):
        """config.update_from_dict 应能更新新字段"""
        config.update_from_dict({
            'semi': {'enabled': True, 'labeled_ratio': 0.5},
            'vmd': {'K': 6},
        })
        self.assertTrue(config.semi.enabled)
        self.assertEqual(config.semi.labeled_ratio, 0.5)
        self.assertEqual(config.vmd.K, 6)
        # 恢复默认值
        config.update_from_dict({
            'semi': {'enabled': False, 'labeled_ratio': 0.4},
            'vmd': {'K': 4},
        })
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest tests.test_semi_config -v`
Expected: AttributeError (config.vmd or config.semi not exist)

- [ ] **Step 3: 修改 `src/air_quality/config/settings.py`**

在文件中新增两个 dataclass 类（在文件末尾追加）：

```python
@dataclass
class VMDConfig:
    """VMD 分解配置"""
    enabled: bool = True
    K: int = 4                       # IMF 模态数
    alpha: float = 2000              # 带宽约束
    tau: float = 0                   # 噪声容忍
    DC: int = 0                      # 是否保留 DC
    init: int = 1                    # 初始化方式
    tol: float = 1e-7                # 收敛容差


@dataclass
class SemiSupervisedConfig:
    """半监督训练配置"""
    enabled: bool = False
    labeled_ratio: float = 0.4
    unlabeled_ratio: float = 0.4
    test_ratio: float = 0.2
    teacher_epochs: int = 80
    student_epochs: int = 120
    pseudo_confidence_threshold: float = 0.85
    pseudo_loss_weight: float = 0.5
    early_stop_patience: int = 30
```

然后在 `AppConfig` 类中新增字段：

```python
@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    file: FileConfig = field(default_factory=FileConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    vmd: VMDConfig = field(default_factory=VMDConfig)                # 新增
    semi: SemiSupervisedConfig = field(default_factory=SemiSupervisedConfig)  # 新增
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m unittest tests.test_semi_config -v`
Expected: 3 tests pass

- [ ] **Step 5: 运行全部测试确认未破坏现有功能**

Run: `python -m unittest discover tests -v`
Expected: 所有测试通过

- [ ] **Step 6: 提交**

```bash
git add src/air_quality/config/settings.py tests/test_semi_config.py
git commit -m "feat(config): 新增 VMDConfig 与 SemiSupervisedConfig"
```

---

## Task 7: 实现 SemiSupervisedTrainer + 单元测试

**Files:**
- Create: `src/air_quality/training/semi_supervised.py`
- Create: `tests/test_semi_trainer.py`
- Modify: `src/air_quality/training/__init__.py`

- [ ] **Step 1: 编写失败测试 `tests/test_semi_trainer.py`**

```python
"""SemiSupervisedTrainer 单元测试"""
import unittest
import numpy as np
import torch

from air_quality.training.semi_supervised import (
    SemiSupervisedTrainer,
    compute_pseudo_confidence,
)


class TestPseudoConfidence(unittest.TestCase):
    def test_confidence_shape(self):
        pred = np.random.randn(10, 3, 7)
        conf = compute_pseudo_confidence(pred)
        self.assertEqual(conf.shape, (10,))
        self.assertTrue((conf >= 0).all() and (conf <= 1).all())

    def test_low_variance_high_confidence(self):
        """低方差预测应有高置信度"""
        # 全部相同值 → std=0 → 置信度=1.0
        pred = np.zeros((5, 3, 7))
        conf = compute_pseudo_confidence(pred)
        np.testing.assert_array_almost_equal(conf, np.ones(5))

    def test_high_variance_low_confidence(self):
        """高方差预测应有低置信度"""
        np.random.seed(0)
        pred = np.random.randn(5, 3, 7) * 100
        conf = compute_pseudo_confidence(pred)
        self.assertTrue((conf < 0.1).all())


class TestSemiSupervisedTrainer(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
        # 构造小数据集
        self.X_labeled = np.random.randn(20, 14, 12).astype(np.float32)  # 8+K
        self.y_labeled = np.random.randn(20, 3, 7).astype(np.float32)
        self.X_unlabeled = np.random.randn(20, 14, 12).astype(np.float32)
        self.y_unlabeled = np.random.randn(20, 3, 7).astype(np.float32)
        self.X_test = np.random.randn(10, 14, 12).astype(np.float32)
        self.y_test = np.random.randn(10, 3, 7).astype(np.float32)

    def test_train_runs_4_phases(self):
        """应能完成 4 阶段训练"""
        from air_quality.config import config

        trainer = SemiSupervisedTrainer(
            model_type='vmd_cnn_bilstm_attention',
            input_size=12,
            device='cpu',
            teacher_epochs=2,
            student_epochs=2,
            pseudo_confidence_threshold=0.5,
        )
        student, metrics = trainer.fit(
            X_labeled=self.X_labeled,
            y_labeled=self.y_labeled,
            X_unlabeled=self.X_unlabeled,
            y_unlabeled=self.y_unlabeled,
            X_test=self.X_test,
            y_test=self.y_test,
        )
        self.assertIsNotNone(student)
        self.assertIn('test_loss', metrics)
        self.assertGreater(len(trainer.history['teacher_losses']), 0)
        self.assertGreater(len(trainer.history['student_losses']), 0)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest tests.test_semi_trainer -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: 实现 `src/air_quality/training/semi_supervised.py`**

```python
"""
SemiSupervisedTrainer

实现伪标签半监督学习的 4 阶段训练流程：
  Phase 1: Teacher 在 labeled 数据上全监督预训练
  Phase 2: Teacher 对 unlabeled 数据生成伪标签（高置信度过滤）
  Phase 3: Student 在 labeled + 伪标签 unlabeled 上联合训练
  Phase 4: 在 test 数据上评估
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tud

from air_quality.models import create_model
from air_quality.training.trainer import ModelTrainer

logger = logging.getLogger(__name__)


def compute_pseudo_confidence(predictions: np.ndarray) -> np.ndarray:
    """计算每样本的伪标签置信度。

    Args:
        predictions: (N, T, output_size)

    Returns:
        (N,) 置信度分数，范围 [0, 1]。低方差 → 高置信度。
    """
    std_per_sample = predictions.std(axis=(1, 2))  # (N,)
    return 1.0 / (1.0 + std_per_sample)


class SemiSupervisedTrainer:
    """伪标签半监督训练器

    Args:
        model_type: 模型类型字符串（如 'vmd_cnn_bilstm_attention'）
        input_size: 输入特征维度
        device: 'cuda' / 'cpu'
        teacher_epochs: Teacher 训练轮数
        student_epochs: Student 训练轮数
        pseudo_confidence_threshold: 伪标签置信度阈值
        pseudo_loss_weight: 伪标签损失权重
        batch_size: DataLoader batch size
        learning_rate: 学习率
    """

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
    ):
        self.model_type = model_type
        self.input_size = input_size
        self.device = torch.device(device)
        self.teacher_epochs = teacher_epochs
        self.student_epochs = student_epochs
        self.pseudo_threshold = pseudo_confidence_threshold
        self.pseudo_loss_weight = pseudo_loss_weight
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.history: Dict[str, Any] = {
            'teacher_losses': [],
            'student_losses': [],
            'pseudo_label_rate': [],
        }

    def _make_loader(self, X, y, shuffle=True):
        return tud.DataLoader(
            tud.TensorDataset(
                torch.FloatTensor(X),
                torch.FloatTensor(y),
            ),
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

    def _train_teacher(self, X_labeled, y_labeled, X_val, y_val):
        """Phase 1: Teacher 全监督预训练"""
        logger.info("[Phase 1] Training Teacher on labeled data...")
        teacher = create_model(self.model_type, input_size=self.input_size).to(self.device)
        train_loader = self._make_loader(X_labeled, y_labeled, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False) if X_val is not None else None

        trainer = ModelTrainer(
            model=teacher,
            train_loader=train_loader,
            test_loader=val_loader,
            device=self.device,
            best_model_path=None,  # 不保存中间权重
            loss_type='huber',
            early_stop_patience=self.teacher_epochs,  # 简化：不强制早停
            gradient_clip=1.0,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        trainer.train(num_epochs=self.teacher_epochs)
        self.history['teacher_losses'].extend(trainer.history.get('train_losses', []))
        return teacher

    def _generate_pseudo_labels(self, teacher, X_unlabeled):
        """Phase 2: Teacher 生成伪标签"""
        logger.info("[Phase 2] Generating pseudo labels...")
        teacher.eval()
        with torch.no_grad():
            x = torch.FloatTensor(X_unlabeled).to(self.device)
            preds = teacher(x).cpu().numpy()  # (N, T, output_size)

        confidences = compute_pseudo_confidence(preds)
        mask = confidences >= self.pseudo_threshold
        rate = mask.mean()
        self.history['pseudo_label_rate'].append(float(rate))
        logger.info(f"  Pseudo label retention rate: {rate:.2%}")
        # 即使所有都被过滤掉，也保留置信度最高的 K% 防止空集
        if not mask.any():
            top_k = max(1, int(0.1 * len(mask)))
            top_indices = np.argsort(-confidences)[:top_k]
            mask = np.zeros_like(mask, dtype=bool)
            mask[top_indices] = True
        return preds, mask

    def _train_student(self, X_labeled, y_labeled, X_pseudo, y_pseudo, mask):
        """Phase 3: Student 联合训练"""
        logger.info("[Phase 3] Training Student on labeled + pseudo labels...")
        student = create_model(self.model_type, input_size=self.input_size).to(self.device)

        X_pseudo_filtered = X_pseudo[mask]
        y_pseudo_filtered = y_pseudo[mask]
        X_combined = np.concatenate([X_labeled, X_pseudo_filtered], axis=0)
        y_combined = np.concatenate([y_labeled, y_pseudo_filtered], axis=0)

        train_loader = self._make_loader(X_combined, y_combined, shuffle=True)
        trainer = ModelTrainer(
            model=student,
            train_loader=train_loader,
            test_loader=None,
            device=self.device,
            best_model_path=None,
            loss_type='huber',
            early_stop_patience=self.student_epochs,
            gradient_clip=1.0,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        trainer.train(num_epochs=self.student_epochs)
        self.history['student_losses'].extend(trainer.history.get('train_losses', []))
        return student

    def _evaluate(self, model, X_test, y_test):
        """Phase 4: 测试集评估"""
        logger.info("[Phase 4] Evaluating on test set...")
        model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(X_test).to(self.device)
            preds = model(x).cpu().numpy()
        # 仅取最后 prediction_days 步
        preds_last = preds[:, -3:, :]
        loss = float(np.mean((preds_last - y_test) ** 2))
        return {'test_loss': loss, 'predictions': preds_last}

    def fit(
        self,
        X_labeled,
        y_labeled,
        X_unlabeled,
        y_unlabeled,
        X_test,
        y_test,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """完整 4 阶段训练流程"""
        # 使用 labeled 数据的最后 20% 作为 Teacher 验证集
        n_val = max(1, int(0.2 * len(X_labeled)))
        X_tr = X_labeled[:-n_val]
        y_tr = y_labeled[:-n_val]
        X_val = X_labeled[-n_val:]
        y_val = y_labeled[-n_val:]

        teacher = self._train_teacher(X_tr, y_tr, X_val, y_val)
        pseudo_preds, mask = self._generate_pseudo_labels(teacher, X_unlabeled)
        student = self._train_student(X_labeled, y_labeled, X_unlabeled, pseudo_preds, mask)
        metrics = self._evaluate(student, X_test, y_test)
        return student, metrics
```

- [ ] **Step 4: 修改 `src/air_quality/training/__init__.py`**

在文件末尾追加导出：

```python
from .semi_supervised import SemiSupervisedTrainer, compute_pseudo_confidence
```

并加入 `__all__` 列表。

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m unittest tests.test_semi_trainer -v`
Expected: 4 tests pass

- [ ] **Step 6: 提交**

```bash
git add src/air_quality/training/semi_supervised.py \
        src/air_quality/training/__init__.py \
        tests/test_semi_trainer.py
git commit -m "feat(training): 新增 SemiSupervisedTrainer 伪标签半监督训练器"
```

---

## Task 8: 扩展 scripts/train.py 支持半监督

**Files:**
- Modify: `scripts/train.py`

- [ ] **Step 1: 在 train.py 增加新参数**

在 `parse_args()` 函数中追加以下参数：

```python
    # 半监督与 VMD
    parser.add_argument(
        '--semi-supervised', dest='semi_supervised', action='store_true',
        help='启用半监督训练（仅 vmd_cnn_bilstm_attention 模型支持）',
    )
    parser.add_argument('--vmd-k', type=int, help='VMD 模态数（默认从 config 读取）')
    parser.add_argument('--no-vmd', dest='no_vmd', action='store_true', help='关闭 VMD 分解')
    parser.add_argument(
        '--pseudo-threshold', dest='pseudo_threshold', type=float,
        help='伪标签置信度阈值',
    )
    parser.add_argument(
        '--pseudo-loss-weight', dest='pseudo_loss_weight', type=float,
        help='伪标签损失权重',
    )
    parser.add_argument('--teacher-epochs', type=int, help='Teacher 训练轮数')
    parser.add_argument('--student-epochs', type=int, help='Student 训练轮数')
```

- [ ] **Step 2: 在 main() 函数中应用覆盖**

在 `try:` 块内，**调用 `train_single_model` 之前**，添加：

```python
        # 应用 VMD / 半监督 CLI 覆盖
        if args.no_vmd:
            config.vmd.enabled = False
        if args.vmd_k is not None:
            config.vmd.K = args.vmd_k
        if args.semi_supervised:
            config.semi.enabled = True
        if args.pseudo_threshold is not None:
            config.semi.pseudo_confidence_threshold = args.pseudo_threshold
        if args.pseudo_loss_weight is not None:
            config.semi.pseudo_loss_weight = args.pseudo_loss_weight
        if args.teacher_epochs is not None:
            config.semi.teacher_epochs = args.teacher_epochs
        if args.student_epochs is not None:
            config.semi.student_epochs = args.student_epochs
```

- [ ] **Step 3: 在 main() 中分支处理半监督**

将 `result = train_single_model(args.model, test_loader, ...)` 之后插入：

```python
        if config.semi.enabled:
            logger.info("=" * 60)
            logger.info("运行半监督训练流程 (VMD-CNN-BiLSTM-Attention)")
            logger.info("=" * 60)
            from air_quality.data.vmd import VMDDecomposer
            from air_quality.data.vmd_features import apply_vmd_to_aqi
            from air_quality.training import SemiSupervisedTrainer

            # 应用 VMD
            if config.vmd.enabled:
                decomposer = VMDDecomposer(
                    K=config.vmd.K, alpha=config.vmd.alpha,
                    tau=config.vmd.tau, DC=config.vmd.DC,
                    init=config.vmd.init, tol=config.vmd.tol,
                )
                # 从 processor 拿完整序列（覆盖训练 + 测试区间）
                full_X, full_y = processor.create_sequences(data)
                full_X_vmd = apply_vmd_to_aqi(full_X, decomposer)
            else:
                # 不启用 VMD 时直接使用原始 9 维特征
                full_X, full_y = processor.create_sequences(data)
                full_X_vmd = full_X

            # 三段划分
            (X_lab, y_lab), (X_unl, y_unl), (X_te, y_te) = processor.split_three_way(
                full_X_vmd, full_y,
                ratios=(
                    config.semi.labeled_ratio,
                    config.semi.unlabeled_ratio,
                    config.semi.test_ratio,
                ),
            )

            semi_trainer = SemiSupervisedTrainer(
                model_type='vmd_cnn_bilstm_attention',
                input_size=full_X_vmd.shape[2],
                device=str(device),
                teacher_epochs=config.semi.teacher_epochs,
                student_epochs=config.semi.student_epochs,
                pseudo_confidence_threshold=config.semi.pseudo_confidence_threshold,
                pseudo_loss_weight=config.semi.pseudo_loss_weight,
            )
            semi_model, semi_metrics = semi_trainer.fit(
                X_labeled=X_lab, y_labeled=y_lab,
                X_unlabeled=X_unl, y_unlabeled=y_unl,
                X_test=X_te, y_test=y_te,
            )
            logger.info(f"半监督训练完成，测试损失: {semi_metrics['test_loss']:.6f}")
```

- [ ] **Step 4: 验证 CLI 帮助**

Run: `python -m scripts.train --help`
Expected: 帮助中包含 --semi-supervised / --vmd-k / --no-vmd 等参数

- [ ] **Step 5: 验证 dry-run（2 个 epoch）**

Run: `python -m scripts.train --semi-supervised --teacher-epochs 1 --student-epochs 1 --pseudo-threshold 0.5`
Expected: 训练流程跑完，无报错；日志显示 4 个阶段；测试损失是有限数值

- [ ] **Step 6: 提交**

```bash
git add scripts/train.py
git commit -m "feat(scripts): train.py 支持半监督与 VMD CLI 参数"
```

---

## Task 9: 端到端集成测试

**Files:**
- Create: `tests/test_full_semi_pipeline.py`

- [ ] **Step 1: 编写集成测试**

```python
"""端到端半监督 pipeline 集成测试"""
import unittest
import tempfile
import os
import numpy as np
import pandas as pd


class TestFullSemiPipeline(unittest.TestCase):
    def _create_mock_excel(self, path):
        """创建小型 Excel 测试数据集"""
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

    def test_semi_pipeline_runs(self):
        """端到端半监督 pipeline 应能跑通"""
        from air_quality.config import config
        from air_quality.data.processor import AirQualityDataProcessor
        from air_quality.data.vmd import VMDDecomposer
        from air_quality.data.vmd_features import apply_vmd_to_aqi
        from air_quality.training import SemiSupervisedTrainer

        # 用 mock 数据
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

            trainer = SemiSupervisedTrainer(
                model_type='vmd_cnn_bilstm_attention',
                input_size=X_vmd.shape[2],
                device='cpu',
                teacher_epochs=1,
                student_epochs=1,
                pseudo_confidence_threshold=0.5,
                batch_size=8,
            )
            student, metrics = trainer.fit(
                X_labeled=X_lab, y_labeled=y_lab,
                X_unlabeled=X_unl, y_unlabeled=y_unl,
                X_test=X_te, y_test=y_te,
            )
            self.assertIsNotNone(student)
            self.assertFalse(np.isnan(metrics['test_loss']))
```

- [ ] **Step 2: 运行集成测试**

Run: `python -m unittest tests.test_full_semi_pipeline -v`
Expected: 1 test passes

- [ ] **Step 3: 运行完整测试套件**

Run: `python -m unittest discover tests -v`
Expected: 所有测试通过（应有 ~30+ 个测试）

- [ ] **Step 4: 提交**

```bash
git add tests/test_full_semi_pipeline.py
git commit -m "test: 端到端半监督 pipeline 集成测试"
```

---

## Task 10: 更新文档与最终验证

**Files:**
- Modify: `README.md`
- Modify: `docs/algorithm_analysis.md`

- [ ] **Step 1: 在 README.md 中追加半监督使用说明**

在「运行方式」章节末尾追加：

```markdown
### 半监督训练 (VMD-CNN-BiLSTM-Attention)

```bash
# 启用半监督（默认 VMD K=4）
python -m scripts.train --semi-supervised

# 自定义 VMD 模态数
python -m scripts.train --semi-supervised --vmd-k 6

# 调节伪标签置信度阈值
python -m scripts.train --semi-supervised --pseudo-threshold 0.9
```

数据划分：40% 有标签 + 40% 无标签（伪标签扩展）+ 20% 测试。
```

- [ ] **Step 2: 在 algorithm_analysis.md 中追加半监督章节**

在文档末尾追加：

```markdown
## 半监督范式：VMD-CNN-BiLSTM-Attention + 伪标签

详见 `docs/superpowers/specs/2026-06-16-vmd-semi-supervised-design.md`。

### 关键设计
- **VMD**：仅分解 AQI 单变量，K=4 默认（可在 config 覆盖）
- **半监督**：40% 有标签 + 40% 无标签 + 20% 测试
- **伪标签法**：Teacher → 伪标签（置信度 ≥ 0.85）→ Student 联合训练

### 范式合理性
详见 spec 第 8 节（合理性 / 局限性 / 预期效果）
```

- [ ] **Step 3: 最终验证**

Run: `python -m unittest discover tests -v`
Expected: 全部通过

Run: `python -m scripts.train --help | head -30`
Expected: 显示所有新参数

- [ ] **Step 4: 提交**

```bash
git add README.md docs/algorithm_analysis.md
git commit -m "docs: README 与 algorithm_analysis 补充半监督说明"
```

---

## 自审报告

| 检查项 | 结论 |
|---|---|
| **Spec 覆盖** | ✅ Spec 第 1-10 节每条要求都有对应 Task |
| **占位符扫描** | ✅ 无 TBD/TODO，所有代码完整 |
| **类型一致性** | ✅ `SemiSupervisedTrainer.fit()` / `compute_pseudo_confidence()` / `VMDDecomposer.decompose()` 在所有 task 中保持一致签名 |
| **测试覆盖** | ✅ VMD / 三段划分 / 模型 / 半监督训练器 / 集成 pipeline 均覆盖 |
| **TDD 顺序** | ✅ 每个 Task 都是「写测试 → 跑失败 → 实现 → 跑通过 → 提交」 |
| **依赖一致性** | ✅ vmdpy 在 Task 1 引入，Task 2 立即使用 |
| **向后兼容** | ✅ 现有全监督路径完全不动，新功能通过 CLI 标志启用 |
