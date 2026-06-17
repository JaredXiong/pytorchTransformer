# VMD-CNN-BiLSTM-Attention 半监督范式设计

**日期**：2026-06-16
**作者**：JaredXiong
**状态**：待审阅

## 1. 背景与目标

### 1.1 现状

当前项目基于**全监督范式**：

- 数据流：Excel → 预处理 → 序列构造 → 80/20 划分 → 端到端训练
- 模型：Transformer / LSTM / CNN / Hybrid 四种 backbone
- 训练器：单阶段 `ModelTrainer`，所有训练样本均带真实标签
- 评估：标准 7 维污染物 MSE/RMSE/MAE/R²

### 1.2 目标

将项目从全监督范式迁移至**半监督范式**，引入 VMD-CNN-BiLSTM-Attention 专用架构，验证以下假设：

1. 在 40% 标注稀缺场景下，伪标签半监督能逼近全监督性能（差距 < 15%）
2. VMD 分解 AQI 能提供 5-10% 的指标增益
3. CNN-BiLSTM-Attention 三段式架构在多步空气质量预测上具有竞争力

### 1.3 关键决策（来自 brainstorming 澄清）

| 决策项 | 选择 |
|---|---|
| 数据划分 | 40% 有标签 + 40% 无标签 + 20% 测试（标准半监督） |
| 半监督方法 | 伪标签法（Teacher → 伪标签 → Student 联合训练） |
| VMD 分解范围 | 仅分解 AQI 单变量 |
| VMD 模态数 K | 默认 4，可在 config 覆盖（4/6/8） |

## 2. 架构

### 2.1 模型定义：`VMDCNNBILSTMAttentionModel`

**继承**：`BaseModel`（与现有 transformer/lstm/cnn/hybrid 一致接口）

**数据流**：

```
输入 (B, 14, 9)
   │
   ▼
[VMD 分解] 最后一维 AQI 序列 (B, 14) → K 个 IMF 分量 (B, K, 14)
   │
   ▼
[特征拼接] 原始 8 维（其余 6 污染物 + month + season）+ K 个 IMF
           → (B, 14, 8+K)  按时间步对齐
   │
   ▼
[CNN] 多尺度 Conv1d (kernel=3/5/7, filters=64) + BN + ReLU + MaxPool
       → 局部特征
   │
   ▼
[BiLSTM] 2 层双向 LSTM (hidden=128)
       → 时序依赖
   │
   ▼
[Attention] 时间步加权
       → 关键时间步突出
   │
   ▼
[FC Head] 3×7 = 21 → reshape (B, 3, 7)
```

**关键参数**（默认）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `K` | 4 | IMF 数量 |
| `cnn_filters` | 64 | CNN 卷积核数量 |
| `bilstm_hidden` | 128 | BiLSTM 隐藏层维度 |
| `bilstm_layers` | 2 | BiLSTM 层数 |
| `dropout` | 0.1 | dropout 比率 |

### 2.2 VMD 分解器：`src/air_quality/data/vmd.py`

封装 `vmdpy` 库：

```python
class VMDDecomposer:
    def __init__(self, K=4, alpha=2000, tau=0, DC=0, init=1, tol=1e-7):
        ...

    def decompose(self, signal: np.ndarray) -> np.ndarray:
        """对单变量序列做 VMD 分解，返回 (K, T) 的 IMF 矩阵。"""
        ...
```

**关键约束**：
- 仅作用于 AQI 单变量
- K=4 时每个 IMF 14 个点，对应 14 天窗口
- 重构误差应 < 1e-3（测试断言）

## 3. 数据流改造

### 3.1 三段划分：`split_three_way`

在 `AirQualityDataProcessor` 中新增：

```python
def split_three_way(
    self,
    X: np.ndarray,
    y: np.ndarray,
    ratios: tuple = (0.4, 0.4, 0.2),
) -> tuple:
    """按时间顺序三段划分（不打乱），保持时序连续性。"""
    assert abs(sum(ratios) - 1.0) < 1e-6
    n = len(X)
    n_label = int(n * ratios[0])
    n_unlabel = int(n * ratios[1])
    return (
        (X[:n_label], y[:n_label]),
        (X[n_label:n_label+n_unlabel], y[n_label:n_label+n_unlabel]),
        (X[n_label+n_unlabel:], y[n_label+n_unlabel:]),
    )
```

> **关键约束**：按时间顺序划分，不打乱。空气质量数据具有强时间依赖，随机划分会泄露未来信息。

### 3.2 数据预处理集成

VMD 分解发生在 `create_sequences` 之后、模型输入之前：

```python
X, y = processor.create_sequences(data)               # (N, 14, 9), (N, 3, 7)
X_vmd = apply_vmd_to_aqi(X, decomposer)              # (N, 14, 8+K)
(labeled, unlabel, test) = split_three_way(X_vmd, y)
```

## 4. 训练流程：`SemiSupervisedTrainer`

### 4.1 四阶段伪标签训练

```
┌─────────────────────────────────────────────────────────┐
│ Phase 1: Teacher 预训练 (在 40% labeled 上)              │
│   - 全监督，epochs = config.semi.teacher_epochs (默认80) │
│   - 早停基于 val_loss                                     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 2: 伪标签生成 (对 40% unlabeled)                   │
│   - Teacher 推理 (B, 14, 9) → (B, 3, 7)                │
│   - 计算置信度 = 1 / (1 + std(pred, dim=time))         │
│   - 仅保留置信度 ≥ threshold (默认 0.85) 的样本            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 3: Student 联合训练                                │
│   - 数据 = 40% 真标签 + 40% 高置信伪标签                  │
│   - 损失 = L_supervised(labeled) + λ·L_pseudo(pseudo)   │
│   - λ 默认为 0.5（防止伪标签误差主导）                   │
│   - epochs = config.semi.student_epochs (默认 120)      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 4: 测试集评估 (20% test)                           │
│   - MSE/RMSE/MAE/R² + 可视化 7 子图                      │
└─────────────────────────────────────────────────────────┘
```

### 4.2 关键代码结构

```python
class SemiSupervisedTrainer:
    def __init__(self, model_cls, config, device):
        self.model_cls = model_cls
        self.config = config
        self.device = device

    def fit(self, labeled_data, unlabeled_data, test_data):
        teacher = self._train_teacher(labeled_data)
        pseudo_labels, mask = self._generate_pseudo_labels(teacher, unlabeled_data)
        student = self._train_student(labeled_data, pseudo_labels, mask)
        metrics = self._evaluate(student, test_data)
        return student, metrics

    def _train_teacher(self, labeled_data):
        ...

    def _generate_pseudo_labels(self, model, unlabeled_data):
        ...

    def _train_student(self, labeled, pseudo, mask):
        ...
```

### 4.3 伪标签置信度计算

```python
def compute_confidence(predictions: np.ndarray) -> np.ndarray:
    """predictions: (N, 3, 7)
    返回每样本的置信度分数 (N,)，范围 [0, 1]。
    """
    std_per_sample = predictions.std(axis=(1, 2))  # (N,)
    return 1.0 / (1.0 + std_per_sample)
```

## 5. 配置

### 5.1 新增配置类

```python
@dataclass
class SemiSupervisedConfig:
    enabled: bool = False
    labeled_ratio: float = 0.4
    unlabeled_ratio: float = 0.4
    test_ratio: float = 0.2
    teacher_epochs: int = 80
    student_epochs: int = 120
    pseudo_confidence_threshold: float = 0.85
    pseudo_loss_weight: float = 0.5
    early_stop_patience: int = 30


@dataclass
class VMDConfig:
    enabled: bool = True
    K: int = 4
    alpha: float = 2000
    tau: float = 0
    DC: int = 0
    init: int = 1
    tol: float = 1e-7
```

### 5.2 命令行入口

```bash
# 半监督训练（默认走 VMD-CNN-BiLSTM-Attention）
python -m scripts.train --semi-supervised

# 指定 VMD 模态数
python -m scripts.train --semi-supervised --vmd-k 6

# 关闭 VMD，对比纯半监督
python -m scripts.train --semi-supervised --no-vmd

# 调节置信度阈值
python -m scripts.train --semi-supervised --pseudo-threshold 0.9

# 调节伪标签损失权重
python -m scripts.train --semi-supervised --pseudo-loss-weight 0.3
```

## 6. 文件变更清单

### 6.1 新增文件

```
src/air_quality/
├── models/
│   └── vmd_cnn_bilstm_attention.py     # VMD-CNN-BiLSTM-Attention 模型
├── data/
│   └── vmd.py                            # VMD 分解器
└── training/
    └── semi_supervised.py               # SemiSupervisedTrainer

tests/
├── test_vmd.py                           # VMD 分解单元测试
├── test_semi_supervised.py               # 半监督训练流程测试
├── test_split_three_way.py               # 三段划分测试
├── test_vmd_cnn_bilstm.py               # 模型 forward 测试
└── test_semi_trainer.py                  # 4 阶段训练流程测试
```

### 6.2 修改文件

```
src/air_quality/
├── models/
│   └── factory.py                       # 注册 'vmd_cnn_bilstm_attention'
├── models/
│   └── __init__.py                      # 导出新模型
├── data/
│   └── processor.py                     # 增加 split_three_way
├── config/
│   └── settings.py                      # 新增 SemiSupervisedConfig + VMDConfig
└── scripts/
    └── train.py                         # 增加 --semi-supervised / --vmd-k 等参数

requirements.txt                          # 新增 vmdpy>=0.2
```

## 7. 测试策略

### 7.1 单元测试

| 测试文件 | 测试目标 | 关键断言 |
|---|---|---|
| `test_vmd.py` | VMD 分解器 | K 个 IMF 形状正确、能量守恒、重构误差 < 1e-3 |
| `test_split_three_way.py` | 三段划分 | 比例正确、时序连续、无重叠 |
| `test_semi_supervised.py` | 伪标签生成 | 阈值过滤正确、置信度计算稳定 |
| `test_vmd_cnn_bilstm.py` | 模型 forward | 输出形状 (B, 3, 7)、参数初始化 |
| `test_semi_trainer.py` | 4 阶段训练 | Teacher 收敛、伪标签非空、Student 损失下降 |

### 7.2 集成测试

- `test_full_semi_pipeline.py`：端到端跑 5 epochs，验证模型可保存、指标非 NaN。

### 7.3 评估对比

三组对比实验设计：

| 范式 | 训练数据 | 用途 |
|---|---|---|
| 全监督 baseline (Hybrid) | 80% 全部标签 | 上界参考 |
| 半监督 VMD-CNN-BiLSTM | 40% 标签 + 40% 伪标签 | 主结果 |
| 全监督 VMD-CNN-BiLSTM | 80% 标签 | 隔离 VMD 贡献 |

## 8. 范式合理性评估

### 8.1 合理性 ✅

1. **数据规模匹配**：北京 2015-2024 约 3500+ 天，40% labeled ≈ 1400+ 序列，足够支撑半监督训练。
2. **架构契合**：
   - VMD 把 AQI 分解为趋势/周期/随机分量，符合空气污染多尺度耦合机理
   - CNN 提取 IMF 局部模式
   - BiLSTM 捕获前后文时序依赖
   - Attention 突出关键时间步（如污染爆发前兆）
3. **学术范式**：VMD-CNN-BiLSTM-Attention 是 2022-2024 时间序列预测领域的主流架构。
4. **半监督必要性**：空气质量数据获取成本高（部分时段缺失传感器），半监督可缓解标注稀缺。

### 8.2 局限性 ⚠

1. **VMD 边缘效应**：序列两端会出现端点失真，14 天窗口相对短，影响有限。
2. **伪标签误差传播**：Student 继承 Teacher 偏差，需通过置信度阈值 + 低权重缓解。
3. **计算成本**：VMD 增加预处理时间，但 K=4 时增量很小。
4. **可解释性**：VMD 分解的物理意义不如 EMD 明确。

### 8.3 预期效果

- 半监督应 **接近** 全监督性能（差距 < 15%）
- VMD 贡献：**5-10%** 指标提升（与基线 hybrid 对比）
- 伪标签过滤应提升稳定性，减少过拟合

## 9. 实施步骤（高层）

1. 引入 `vmdpy` 依赖
2. 实现 `VMDDecomposer` + 单元测试
3. 实现 `split_three_way` + 单元测试
4. 实现 `VMDCNNBILSTMAttentionModel` + 单元测试
5. 实现 `SemiSupervisedTrainer` + 单元测试
6. 扩展 `config/settings.py` 与 `scripts/train.py` CLI
7. 在 `factory.py` 与 `models/__init__.py` 注册新模型
8. 端到端集成测试
9. 编写评估对比脚本
10. 更新 `docs/algorithm_analysis.md`

## 10. 风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| vmdpy 与 Python 版本不兼容 | 中 | 锁定 Python 3.9-3.11 |
| 半监督效果差于全监督 | 低 | 保留全监督基线，必要时回退 |
| 训练时间过长 | 低 | K=4 + Teacher/Student 共用模型 |
| VMD 分解不稳定 | 中 | 单元测试覆盖重构误差 |
| 现有测试被破坏 | 中 | 保持向后兼容，新增独立测试 |
