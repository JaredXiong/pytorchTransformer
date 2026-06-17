# VMD-CNN-BiLSTM-Attention 预训练-微调范式设计

**日期**：2026-06-17
**作者**：JaredXiong
**状态**：待审阅

## 1. 背景与目标

### 1.1 现状

项目当前已有两段式半监督范式(参考 `2026-06-16-vmd-semi-supervised-design.md`):

- 数据流：Excel → 预处理 → VMD 分解 AQI → 序列构造 → 40/40/20 划分 → Teacher/Student 联合训练
- 模型：VMD-CNN-BiLSTM-Attention 共享 backbone
- 训练：4 阶段(Teacher 预训练 → 伪标签 → Student 联合 → 评估)
- 验证：18+ 个单元测试 + 1 个端到端集成测试

### 1.2 目标

在现有两段式半监督基础上,**新增第三层「无监督预训练 + 有监督微调」**:

1. 在 unlabeled 段上做掩码自监督预训练,让 backbone 提前学到 IMF 推演能力
2. 用预训练权重初始化 Teacher 与 Student 的 backbone,缩短有监督阶段的收敛时间
3. 保留现有全监督与半监督路径,新功能通过 CLI 开关 `--pretrain` 启用

### 1.3 关键决策(来自 brainstorming 澄清)

| 决策项 | 选择 |
|---|---|
| VMD 主线 | 沿用现有实现(只分解 AQI,K=4 默认) |
| 预训练任务 | VMD 分量预测式(预测被遮盖的 IMF) |
| 预训练数据 | 仅用 40% unlabeled 段 |
| 预训练 backbone | 复用 VMD-CNN-BiLSTM-Attention 编码器 |
| 预训练 head | 轻量级 FC,接在 Attention 后,预测 K 个 IMF |
| 微调 head | 沿用现有 fc1+fc2,预测 7 维污染物 |
| 与半监督整合 | 三阶段:预训练 → 伪标签 → 微调 |
| 兼容性 | 现有调用方零改动,`forward(mode='finetune')` 默认 |

## 2. 架构

### 2.1 三阶段流水线

```
┌──────────────────────────────────────────────────────────────┐
│ 数据流(40/40/20 切分,沿用 split_three_way)                   │
│   X_unlabeled: (N_un, 14, 8+K)  [VMD 拼接后]                  │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ Phase 0: 预训练(仅用 unlabeled 段)                            │
│   - 任务: 掩码自监督 → 预测被遮盖的 IMF                       │
│   - 模型: backbone(PretrainEncoder) = 现有 VMD-CNN-BiLSTM-Att │
│     + PretrainHead(轻量级 FC)输出被遮盖的 IMF                  │
│   - 完成后: 保留 backbone 权重,丢弃 PretrainHead              │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ Phase 1-3: 现有伪标签半监督(Teacher/Student/伪标签)           │
│   - Teacher 权重从预训练 backbone 初始化                      │
│   - Teacher = backbone + 下游 head (FC → 7)                   │
│   - Student 同上,联合训练(40% labeled + 40% 伪标签)           │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ Phase 4: 测试集评估(沿用)                                    │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 模型拆分

将现有 `VMDCNNBILSTMAttentionModel` 拆为三段:

| 段 | 子模块 | 输出维度 |
|---|---|---|
| 共享 backbone | `input_proj` + `conv3/5/7` + `bn` + `bilstm` + `attention` | `(B, T, bilstm_hidden*2)` |
| 预训练 head | `pretrain_head = nn.Linear(bilstm_hidden*2, vmd_K)` | `(B, T, K)` |
| 微调 head | `finetune_head = nn.Sequential(Linear→ReLU→Dropout→Linear)` | `(B, T, 7)` |

`forward(x, mode='finetune')`:
- `mode='pretrain'`: backbone → pretrain_head → `(B, T, K)`
- `mode='finetune'`: backbone → finetune_head → `(B, T, 7)`(默认,沿用现有语义)

**关键约束**:现有调用方完全不需要改,所有现有测试不破。

### 2.3 预训练器 `Pretrainer`

封装在 `src/air_quality/training/pretrain.py`:

```python
class Pretrainer:
    def __init__(
        self,
        model: VMDCNNBILSTMAttentionModel,
        device: torch.device,
        learning_rate: float = 5e-4,
        weight_decay: float = 3e-4,
        mask_ratio: float = 0.3,
        early_stop_patience: int = 20,
    ): ...

    def fit(self, X_unlabeled: np.ndarray, epochs: int) -> dict:
        """返回 history 字典,包含 pretrain_losses 列表"""

    @staticmethod
    def extract_backbone_state(model: nn.Module) -> dict:
        """提取 backbone 权重,丢弃 pretrain_head"""

    @staticmethod
    def generate_mask(shape: tuple, ratio: float, generator=None) -> np.ndarray:
        """生成 (B, T) 的二值掩码,1=保留,0=遮盖"""
```

### 2.4 三阶段编排器 `PretrainFinetuneTrainer`

继承 `SemiSupervisedTrainer`,在 `fit()` 前增加 Phase 0:

```python
class PretrainFinetuneTrainer(SemiSupervisedTrainer):
    def __init__(self, pretrain_config: PretrainConfig, **kwargs):
        super().__init__(**kwargs)
        self.pretrain_config = pretrain_config

    def fit(self, X_labeled, y_labeled, X_unlabeled, y_unlabeled, X_test, y_test):
        # Phase 0: 预训练(仅 unlabeled)
        if self.pretrain_config.enabled:
            pretrain_model = self._build_model().to(self.device)
            pretrainer = Pretrainer(pretrain_model, self.device,
                                    learning_rate=self.pretrain_config.learning_rate,
                                    weight_decay=self.pretrain_config.weight_decay,
                                    mask_ratio=self.pretrain_config.mask_ratio,
                                    early_stop_patience=self.pretrain_config.early_stop_patience)
            pretrainer.fit(X_unlabeled, epochs=self.pretrain_config.epochs)
            self.history['pretrain_losses'] = pretrainer.history['train_losses']
            backbone_state = Pretrainer.extract_backbone_state(pretrain_model)
        else:
            backbone_state = None

        # Phase 1-4: 沿用父类,但 Teacher/Student 用 backbone_state 初始化
        return self._fit_semi(X_labeled, y_labeled, X_unlabeled, y_unlabeled,
                              X_test, y_test, backbone_state=backbone_state)

    def _fit_semi(self, ..., backbone_state=None):
        # 与父类 fit 等价,但在创建 teacher / student 时若 backbone_state 不为空,
        # 用 model.load_state_dict(backbone_state, strict=False) 加载
        ...
```

## 3. 数据流与算法

### 3.1 预训练数据流

```
X_unlabeled: (N_un, 14, 8+K)
   │ 列:[IMF_0, IMF_1, ..., IMF_{K-1}, pm2.5, pm10, no2, so2, co, o3, month, season]
   ↓
[掩码生成] 对每个样本随机选 1 个时间步的 IMF 列(共 K 个 IMF 之一)置零
   - 简化版"列掩码":shape (N_un, 14, 8+K),值为 0/1
   - 目标:被遮盖的 IMF 列在 14 个时间步上的完整值
   ↓
[前向] backbone → pretrain_head → (N_un, 14, K)  [预测所有 IMF]
   ↓
[损失] MSE(pred, x[:, :, :K]) 仅在被遮盖的 (time_step, imf_index) 计算
   - 其余位置加权 0(避免噪声)
   ↓
[反向] Adam(lr=5e-4) + grad-clip 1.0 + 早停(patience=20)
```

**为什么不预测其他 8 维特征**:
- 你选择"VMD 分量预测式",核心信号是 IMF
- 强制 backbone 学到"通过上下文推 IMF"的能力
- 减少损失噪声(8 维其他特征已被 scaler 标准化,与下游监督重叠)

### 3.2 微调数据流

```
[Teacher 阶段] labeled: (N_lab, 14, 8+K) → y: (N_lab, 3, 7)
   - backbone 加载预训练权重 + head 随机初始化
   - 训练 N epochs,保存最优

[伪标签阶段] Teacher 推理 unlabeled → (N_un, 3, 7) 伪标签
   - 置信度 = 1 / (1 + std)
   - 阈值过滤(默认 0.85)

[Student 阶段] labeled + filtered_pseudo
   - 同 backbone 起点 + 独立 head 随机种子
   - 联合训练,huber 损失

[评估] 20% test
   - MSE / RMSE / MAE / R²
```

### 3.3 Teacher 与 Student 共享起点策略

| 模型 | backbone 初始化 | head 初始化 | 训练模式 |
|---|---|---|---|
| Teacher | 预训练权重(`load_state_dict(strict=False)`) | `torch.manual_seed(seed).normal_()` | 全解冻,有监督 |
| Student | 同 Teacher 的 backbone 权重 | 同 seed(与 Teacher 一致) | 全解冻,联合 |

**为什么共享 backbone 起点**:
- 你选 C(预训练-微调)是为了最大化预训练价值
- 若 Teacher 从零开始,预训练 backbone 的优势在 Teacher 阶段被洗掉
- 共享起点 + 不同训练数据 → 集成效应最大化

## 4. 配置

### 4.1 新增 `PretrainConfig`

```python
@dataclass
class PretrainConfig:
    """无监督预训练配置"""
    enabled: bool = False
    epochs: int = 60
    mask_ratio: float = 0.3          # 0~1,被遮盖时间步的比例
    learning_rate: float = 5e-4
    weight_decay: float = 3e-4
    batch_size: int = 32
    early_stop_patience: int = 20
    target: str = 'imf'             # 固定,只预测 K 个 IMF
```

并在 `AppConfig` 中加:
```python
pretrain: PretrainConfig = field(default_factory=PretrainConfig)
```

### 4.2 CLI 参数

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--pretrain` | 启用预训练-微调范式 | False |
| `--pretrain-epochs` | 预训练轮数 | 60 |
| `--pretrain-mask-ratio` | 掩码比例 | 0.3 |
| `--pretrain-lr` | 预训练学习率 | 5e-4 |

`--pretrain` 只在 `--semi-supervised` 时生效;两个都关闭时走纯全监督路径(沿用)。

### 4.3 模型超参(沿用现有)

| 参数 | 默认值 |
|---|---|
| `vmd_K` | 4 |
| `cnn_filters` | 64 |
| `bilstm_hidden` | 128 |
| `bilstm_layers` | 2 |
| `dropout` | 0.1 |

## 5. 文件变更清单

### 5.1 新增文件

```
src/air_quality/
├── training/
│   ├── pretrain.py                     # 掩码自监督预训练器
│   └── pretrain_finetune.py            # 三阶段编排器

tests/
├── test_pretrain.py                    # 掩码/损失/反向传播
├── test_pretrain_finetune.py           # 三阶段流程
└── test_pretrain_semi_pipeline.py      # 端到端集成
```

### 5.2 修改文件

```
src/air_quality/
├── models/
│   └── vmd_cnn_bilstm_attention.py     # 拆分 backbone / pretrain_head / finetune_head
├── config/
│   └── settings.py                     # 加 PretrainConfig
├── training/
│   ├── __init__.py                     # 导出 Pretrainer / PretrainFinetuneTrainer
│   └── semi_supervised.py              # 提取出可复用的 _build_model_with_state
├── scripts/
│   └── train.py                        # 加 --pretrain 系列参数
└── air_quality/__init__.py             # 顶层导出
```

### 5.3 模型结构调整要点

```python
class VMDCNNBILSTMAttentionModel(BaseModel):
    def __init__(self, ..., vmd_K=4, ...):
        # 共享部分(沿用)
        self.input_proj = nn.Linear(input_size, cnn_filters)
        self.conv3/5/7, self.bn, self.relu
        self.bilstm = nn.LSTM(cnn_filters*3, bilstm_hidden, bilstm_layers, ...)
        self.attention = TemporalAttention(bilstm_hidden*2)

        # 三个 head
        self.pretrain_head = nn.Linear(bilstm_hidden*2, vmd_K)   # 预测 IMF
        self.finetune_head = nn.Sequential(                       # 预测污染物
            nn.Linear(bilstm_hidden*2, bilstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bilstm_hidden, output_size),
        )

    def forward(self, x, mode='finetune'):
        h = self.input_proj(x)
        # ... CNN + BiLSTM + Attention 同前 ...
        if mode == 'pretrain':
            return self.pretrain_head(attended)  # (B, T, K)
        else:
            return self.finetune_head(attended)  # (B, T, 7) 默认
```

**兼容性保证**: `forward(x)` 不带 `mode` 时默认 `finetune`,输出 `(B, T, 7)`,与现有完全一致。

## 6. 测试策略

### 6.1 单元测试

| 测试文件 | 关键测试 | 断言 |
|---|---|---|
| `tests/test_pretrain.py` | 掩码生成正确性 | `mask.sum() ≈ ratio * N * T`;遮盖位置为 0 |
| | 预训练 forward 形状 | `(B, 14, 8+K)` → `(B, 14, K)` |
| | 预训练损失仅在遮盖位置计算 | 验证损失张量只对命中位置非零 |
| | 反向传播梯度流 | `pretrain_head.weight.grad` 与 `bilstm.weight_hh.grad` 都非零 |
| | 多组 mask_ratio(0.1/0.3/0.5) | 形状不变、损失为有限数 |
| `tests/test_pretrain_finetune.py` | 三阶段流水线跑通 | 预训练 → backbone 抽取 → Teacher 训 → Student 训 → 评估 |
| | backbone 权重被正确加载到 Teacher/Student | `torch.equal(teacher.input_proj.weight, student.input_proj.weight)` |
| | Teacher 与 Student 起点一致 | 训练前 `torch.equal(teacher.backbone, student.backbone)` |
| | Teacher 与 Student head 独立 | `not torch.equal(teacher.finetune_head, student.finetune_head)` |
| | 三段日志非空 | `pretrain_losses` / `teacher_losses` / `student_losses` 都有 |
| | `--pretrain` 关闭时退化 | 不走预训练,backbone 不被加载 |
| `tests/test_pretrain_semi_pipeline.py` | mock Excel → VMD → split → pretrain → finetune → test | 端到端 |
| | 指标非 NaN | `metrics['test_loss']` 为有限 float |

### 6.2 回归保证

- 现有 `test_vmd_cnn_bilstm.py`:仅 `model(x)` 模式输出 `(B, 14, 7)` — 保留
- 现有 `test_full_semi_pipeline.py`:走 `--no-pretrain` 路径
- 计划 `python -m unittest discover tests -v`,总测试数从 30+ 增至 **40+**

### 6.3 评估对比(手动实验)

| 范式 | 训练数据 | 用途 |
|---|---|---|
| 全监督 Hybrid baseline | 80% 全部标签 | 上界 |
| 半监督 VMD-CNN-BiLSTM(无预训练) | 40% 标签 + 40% 伪标签 | 隔离"半监督"贡献 |
| 半监督 VMD-CNN-BiLSTM + 预训练 | 同上 + 预训练 backbone 初始化 | 主结果 |

## 7. 风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| backbone 加载 key 不匹配 | 中 | `extract_backbone_state` 只取非 head 的 key(input_proj / conv3/5/7 / bn / bilstm / attention),`load_state_dict(strict=False)` 忽略 fc / pretrain_head 不匹配 |
| 预训练过拟合 unlabeled | 中 | 早停(patience=20)+ dropout 0.1 |
| 三阶段状态污染 | 低 | Teacher/Student 各自新 optimizer,严格隔离 |
| 现有调用方破坏 | 低 | `forward(mode='finetune')` 默认值,行为不变 |
| 预训练-微调效果不显著 | 中 | 提供 `--no-pretrain` 回退路径,纳入对比实验 |

## 8. 实施步骤(高层)

1. 重构 `VMDCNNBILSTMAttentionModel`,拆分 backbone / pretrain_head / finetune_head
2. 实现 `Pretrainer` + 单元测试
3. 实现 `PretrainFinetuneTrainer` + 单元测试
4. 扩展 `config/settings.py` 与 `scripts/train.py` CLI
5. 在 `training/__init__.py` 与 `air_quality/__init__.py` 导出
6. 端到端集成测试
7. 写盘实施计划并执行

## 9. 预期效果

- 预训练-微调应**缩短** Teacher/Student 收敛轮数(预训练 backbone 已学 IMF 推演)
- 测试集 R² 应与现有半监督持平或略高
- 训练时间增加约 30%(预训练阶段)
- 现有全监督 baseline 仍是上界参考
