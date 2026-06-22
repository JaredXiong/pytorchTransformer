# 模型可调参数详解

> 本文档系统梳理 `pytorchTransformer` 项目中每个模型的可调参数、影响范围及代码标注位置。
> 所有引用使用 `文件:行号` 格式，便于在 IDE 中跳转。

---

## 一、全局配置入口（所有模型共享）

`src/air_quality/config/settings.py` 中的 `ModelConfig` dataclass 集中保存了 5 个核心模型参数：

| 参数 | 默认值 | 影响 | 代码位置 |
|---|---|---|---|
| `input_size` | 9 | 输入特征数（7 污染物 + month + season） | `settings.py:11` |
| `output_size` | 7 | 输出维度（仅污染物） | `settings.py:17` |
| `d_model` | 128 | Transformer / 混合模型 / BiLSTM 共享隐藏维度 | `settings.py:12` |
| `lstm_hidden` | 256 | LSTM 隐藏维度 | `settings.py:13` |
| `cnn_filters` | 128 | CNN 卷积核数 | `settings.py:14` |
| `nhead` | 4 | 多头注意力的头数 | `settings.py:15` |
| `num_layers` | 4 | Transformer 编码层数 / LSTM 层数 | `settings.py:16` |
| `dropout` | 0.1 | 全模型通用 dropout（已从 0.2 调低以减弱均值回归） | `settings.py:18` |

可以通过 `config.update_from_dict({'model': {'d_model': 256}})` 运行时覆盖。

---

## 二、各模型独有参数详解

### 1. Transformer (`TransformerModel`)

文件：`src/air_quality/models/transformer.py`

| 参数 | 默认 | 影响 | 代码位置 |
|---|---|---|---|
| `d_model` | 128 | 注意力维度；越大表达力越强但越慢 | `transformer.py:22` |
| `nhead` | 4 | 头数必须能整除 d_model；头数越多捕获子空间越多 | `transformer.py:23` |
| `num_layers` | 4 | 编码器层数；越深表达力越强，训练越慢且更难收敛 | `transformer.py:23` |
| `dropout` | 0.1 | 防过拟合 | `transformer.py:23` |
| `residual_weight` | 0.3 | **残差 shortcut 初始权重**（sigmoid 后 ≈ 0.574）。越大模型越依赖 input projection 直通；越小越依赖 Transformer 输出。可调以平衡"反均值回归"与"长依赖" | `transformer.py:24, 39` |
| `feature_scale_init` | 1.0 | **逐特征门控初始值**。防止 Linear 把不同污染物压到同一均值；如发现某些特征方差过小可放大 | `transformer.py:24, 41` |
| `ffn_mult` | 4 | FFN 维度倍率（`dim_feedforward = ffn_mult * d_model`）；标准 Transformer 是 4 | `transformer.py:25, 48` |
| `norm_first` | True | **pre-LN 开关**。True 更稳定适合深层；False（post-LN）峰值表达力更强但需 warmup | `transformer.py:25, 51` |

全局对应的反平滑配置（在 `settings.py:19-23`）：

- `transformer_residual_weight`
- `transformer_feature_scale_init`
- `transformer_ffn_mult`
- `transformer_norm_first`

这 4 项是在构造 `TransformerModel` 时传给模型的桥接字段。

---

### 2. LSTM (`LSTMModel`)

文件：`src/air_quality/models/lstm.py`

| 参数 | 默认 | 影响 | 代码位置 |
|---|---|---|---|
| `hidden_size` | 256 | LSTM 隐藏状态维度；时序记忆容量 | `lstm.py:12` |
| `num_layers` | 4 | LSTM 层数；越深梯度越难传播，建议 ≤ 4 | `lstm.py:13` |
| `dropout` | 0.05 | 仅在 num_layers > 1 时生效 | `lstm.py:14, 22` |
| `bidirectional` | True | 是否双向；双向参数量翻倍但能利用过去+未来 | `lstm.py:14` |

> 注：LSTM 没有专门的反平滑开关，因为 LSTM 本身就有显式记忆单元，不容易退化成"均值预测"。

---

### 3. CNN (`CNNModel`)

文件：`src/air_quality/models/cnn.py`

| 参数 | 默认 | 影响 | 代码位置 |
|---|---|---|---|
| `num_filters` | 64 | 每条分支的卷积核数 | `cnn.py:13` |
| `kernel_sizes` | `[3, 5, 7]` | **多尺度卷积核列表**；加更多尺寸可捕获更广感受野，但 fc1 输入会变多（`total_filters = num_filters * len(kernel_sizes)`） | `cnn.py:14, 17` |
| `dropout` | 0.2 | 防过拟合 | `cnn.py:14` |

CNN 模型把多种 kernel 的特征 concat 后送入 FC；调整 `kernel_sizes` 直接改变 fc1 输入维度。

---

### 4. Hybrid (`HybridModel`)

文件：`src/air_quality/models/hybrid.py`

Hybrid 模型 = 共享 `input_proj` + Transformer 分支 + LSTM 分支 + CNN 分支 + 三个 LightSelfAttention + 6 路 CrossAttentionFusion + 输出 FC。

#### 顶层参数

| 参数 | 默认 | 影响 | 代码位置 |
|---|---|---|---|
| `d_model` | 128 | 三分支统一隐藏维度 | `hybrid.py:152` |
| `nhead` | 4 | Fusion 层多头注意力头数 | `hybrid.py:152` |
| `dropout` | 0.2 | 各分支 dropout | `hybrid.py:152` |
| `residual_weight` | 0.3 | **Fusion 后残差权重**（可学习）；同 Transformer 的 `residual_weight`，控制 `input_proj` 直通强度 | `hybrid.py:175, 191` |

#### 子模块内部硬编码参数

| 子模块 | 硬编码参数 | 影响 | 代码位置 |
|---|---|---|---|
| `TransformerBranch` | `num_layers=2`、`dim_feedforward=256` | 分支深度 / FFN 宽度 | `hybrid.py:75, 79` |
| `LSTMEncoderBranch` | `num_layers=2`、`bidirectional=True` | 分支层数 / 双向性 | `hybrid.py:94` |
| `CNNEncoderBranch` | `num_filters=64`、`kernel_sizes=[3,5,7]` | 多尺度 CNN 容量 | `hybrid.py:119-122` |
| `LightSelfAttention` | `nhead=2` | 三分支各自的注意力头数 | `hybrid.py:162-164` |
| `CrossAttentionFusion` | 6 路交叉注意力头数 = `nhead` | 融合阶段头数 | `hybrid.py:35-40` |

> 全局桥接字段（`settings.py:91-97` 的 `HybridModelConfig`）目前只暴露 `d_model / nhead / dropout / residual_weight`，其余子模块参数需直接改 `hybrid.py`。

---

### 5. VMD-CNN-BiLSTM-Attention（双头）(`VMDCNNBILSTMAttentionModel`)

文件：`src/air_quality/models/vmd_cnn_bilstm_attention.py`

数据流：input → input_proj → 多尺度 CNN（kernel 3/5/7）→ BiLSTM → 时间步注意力 → 双头（`pretrain_head` / `finetune_head`）。

| 参数 | 默认 | 影响 | 代码位置 |
|---|---|---|---|
| `vmd_K` | 4 | 预训练 head 输出的 IMF 模态数（应与 `VMDConfig.K` 一致） | `vmd_cnn_bilstm_attention.py:41` |
| `cnn_filters` | 64 | 多尺度 CNN 卷积核数 | `vmd_cnn_bilstm_attention.py:42, 51-54` |
| `bilstm_hidden` | 128 | BiLSTM 隐藏维度（最终输出 = `bilstm_hidden * 2`） | `vmd_cnn_bilstm_attention.py:43` |
| `bilstm_layers` | 2 | BiLSTM 层数 | `vmd_cnn_bilstm_attention.py:44` |
| `dropout` | 0.1 | BiLSTM 中间层 dropout（仅 `layers>1` 时生效） | `vmd_cnn_bilstm_attention.py:45, 63` |
| `mode` | `'finetune'` | `forward()` 切换 `'pretrain'` / `'finetune'` 双头输出 | `vmd_cnn_bilstm_attention.py:95` |

#### VMD 分解参数（`settings.py:99-108` 的 `VMDConfig`）

| 参数 | 默认 | 影响 | 代码位置 |
|---|---|---|---|
| `enabled` | True | 是否启用 VMD 分解 | `settings.py:102` |
| `K` | 4 | IMF 模态数（必须与模型 `vmd_K` 一致） | `settings.py:103` |
| `alpha` | 2000 | 带宽约束；越大带宽越窄（频率更集中） | `settings.py:104` |
| `tau` | 0 | 噪声容忍；> 0 加入噪声鲁棒性 | `settings.py:105` |
| `DC` | 0 | 是否保留 DC 分量（0/1） | `settings.py:106` |
| `init` | 1 | 初始化方式（0/1/2） | `settings.py:107` |
| `tol` | 1e-7 | 收敛容差 | `settings.py:108` |

---

### 6. Ensemble (`EnsembleModel`)

文件：`src/air_quality/models/ensemble.py`

| 参数 | 默认 | 影响 | 代码位置 |
|---|---|---|---|
| `models` | 必填 | 要融合的子模型列表（每个本身需要可独立预测） | `ensemble.py:12` |
| `weights` | 等权 `1/n` | 各模型的加权系数；可手工指定以突出强模型 | `ensemble.py:12, 15-17` |

> 注意：`EnsembleModel` **没有继承 BaseModel**，因此 `create_model('ensemble', ...)` 工厂不支持它——必须在代码中显式构造。

---

## 三、影响所有模型的"非模型"参数

这些不在模型文件里，但会直接决定训练效果，应当与模型参数一起调。

### 训练参数（`settings.py:26-46` 的 `TrainingConfig`）

| 参数 | 默认 | 影响 | 代码位置 |
|---|---|---|---|
| `epochs` | 250 | 总训练轮数 | `settings.py:29` |
| `batch_size` | 32 | 批大小；越大训练越稳，显存占用越高 | `settings.py:30` |
| `learning_rate` | 0.0003 | Adam 学习率 | `settings.py:31` |
| `weight_decay` | 1e-4 | L2 正则（已从 3e-4 调低，避免压制低幅权重） | `settings.py:32` |
| `gradient_clip` | 1.0 | 梯度裁剪阈值 | `settings.py:33` |
| `early_stop_patience` | 40 | 早停耐心值（已从 60 调低） | `settings.py:34` |
| `loss_type` | `'mse_antismooth'` | 损失函数类型：`huber` / `mse` / `mae` / `smooth_l1` / `mse_antismooth` | `settings.py:35` |
| `lambda_var` | 0.1 | 反平滑损失中方差项权重 | `settings.py:38` |
| `lambda_diff` | 0.05 | 反平滑损失中相邻差分项权重 | `settings.py:39` |
| `tau_var` / `tau_diff` | 0.5 / 0.5 | 反平滑损失的目标值 | `settings.py:40-41` |
| `lambda_warmup_epochs` | 20 | 反平滑项前 20 个 epoch 线性升温 | `settings.py:42` |
| `detect_smoothing` | True | 是否启用反平滑早停信号 | `settings.py:44` |
| `smoothing_threshold` | 0.1 | 反平滑触发阈值 | `settings.py:45` |
| `smoothing_stop_patience` | 15 | 反平滑连续触发多少次后早停 | `settings.py:46` |

### 数据参数（`settings.py:49-67` 的 `DataConfig`）

| 参数 | 默认 | 影响 | 代码位置 |
|---|---|---|---|
| `seq_length` | 14 | 输入窗口长度 | `settings.py:53` |
| `prediction_days` | 3 | 预测天数 | `settings.py:54` |
| `train_split_ratio` | 0.8 | 训练集比例 | `settings.py:55` |
| `rolling_windows` | `[]` | 滚动统计特征窗口（当前为空，禁用） | `settings.py:65` |
| `scaler_on_uncapped` | True | True=scaler 在未裁剪数据上 fit，保留真实动态范围 | `settings.py:67` |

### 半监督参数（`settings.py:111-122` 的 `SemiSupervisedConfig`）

| 参数 | 默认 | 影响 | 代码位置 |
|---|---|---|---|
| `labeled_ratio` / `unlabeled_ratio` / `test_ratio` | 0.4 / 0.4 / 0.2 | 三段数据划分 | `settings.py:115-117` |
| `teacher_epochs` / `student_epochs` | 80 / 120 | Teacher/Student 训练轮数 | `settings.py:118-119` |
| `pseudo_confidence_threshold` | 0.85 | 伪标签置信度阈值 | `settings.py:120` |
| `pseudo_loss_weight` | 0.5 | 伪标签损失权重 | `settings.py:121` |

### 预训练参数（`settings.py:125-135` 的 `PretrainConfig`）

| 参数 | 默认 | 影响 | 代码位置 |
|---|---|---|---|
| `epochs` | 60 | 预训练轮数 | `settings.py:129` |
| `mask_ratio` | 0.3 | 掩码比例 | `settings.py:130` |
| `learning_rate` | 5e-4 | 预训练学习率 | `settings.py:131` |
| `weight_decay` | 3e-4 | 预训练权重衰减 | `settings.py:132` |
| `batch_size` | 32 | 预训练批大小 | `settings.py:133` |
| `early_stop_patience` | 20 | 预训练早停耐心 | `settings.py:134` |
| `target` | `'imf'` | 预训练目标（固定为 IMF） | `settings.py:135` |

### 预测参数（`settings.py:79-87` 的 `PredictionConfig`）

| 参数 | 默认 | 影响 | 代码位置 |
|---|---|---|---|
| `min_prediction_value` / `max_prediction_value` | 0 / 500 | 硬裁剪范围 | `settings.py:82-83` |
| `soft_clip` | True | **软饱和开关**：True 避免硬裁剪二次压缩预测动态范围 | `settings.py:86` |
| `soft_clip_scale` | 20.0 | 软饱和缩放因子 | `settings.py:87` |

---

## 四、参数间的耦合关系（容易踩的坑）

1. **`nhead` 必须整除 `d_model`**：`TransformerModel.__init__`（`transformer.py:23, 46`）直接把 `nhead` 传给 `nn.TransformerEncoderLayer`，如果不相容会运行时崩。
2. **VMD 模型要求 `vmd_K == VMDConfig.K`**：`vmd_cnn_bilstm_attention.py:41` 与 `settings.py:103` 需保持一致，否则 `pretrain_head` 的输出维度对不上。
3. **`dropout` 在 LSTM 中条件生效**：`lstm.py:22` 仅当 `num_layers > 1` 时生效；`num_layers=1` 时强制为 0。
4. **Hybrid 子模块是硬编码**：如果想改 `TransformerBranch.num_layers`（默认 2）或 `CNNEncoderBranch.kernel_sizes`，必须直接改 `hybrid.py`，不能用配置覆盖。
5. **反平滑损失只在 `loss_type='mse_antismooth'` 时启用**：`lambda_var` / `lambda_diff` 等参数在其他 loss 下无效。
6. **`scaler_on_uncapped=False` 会破坏极端事件显示**：`_handle_outliers` 会硬裁剪 AQI≤500、CO≤50，关闭该选项会让 scaler 拟合到被裁剪后的数据。

---

## 五、快速调参入口

- **首选**：`config.update_from_dict({...})` 覆盖任意字段（运行时，无需改源文件）
- **次选**：直接修改 `settings.py` 中相应 dataclass 的默认值
- **脚本参数**：`scripts/train.py` 已支持 `--model / --epochs / --batch-size / --lr`，可继续扩展
- **新增模型参数**：在 `src/air_quality/models/<name>.py` 添加 `__init__` 参数 → 在 `settings.py` 的对应 dataclass 增加字段 → 若用 Hybrid 这类带子模块的模型，需要先改子模块签名再在主模型构造中传入