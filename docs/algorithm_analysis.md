# 空气质量预测系统算法架构分析

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 模型架构详解](#2-模型架构详解)
  - [2.1 Transformer](#21-transformer)
  - [2.2 LSTM](#22-lstm)
  - [2.3 CNN](#23-cnn)
  - [2.4 Hybrid](#24-hybrid)
- [3. 监督学习适配性分析](#3-监督学习适配性分析)
- [4. 准确性提升方案](#4-准确性提升方案)
  - [4.1 推荐新增算法](#41-推荐新增算法)
  - [4.2 特征工程优化](#42-特征工程优化)
  - [4.3 模型融合策略](#43-模型融合策略)
  - [4.4 架构改进方向](#44-架构改进方向)
- [5. 综合评估与建议](#5-综合评估与建议)
- [6. 数据预处理与可视化一致性](#6-数据预处理与可视化一致性)

---

## 1. 项目概述

本项目是一个基于深度学习的空气质量时间序列预测系统，采用监督学习范式。系统接收过去14天的9维气象特征作为输入，预测未来3天的7项污染物浓度指标。

**输入特征（9维）**：

| 序号 | 特征名称 | 说明 |
|------|----------|------|
| 0 | aqi | 空气质量指数 |
| 1 | pm2_5_24h | 细颗粒物24小时均值 |
| 2 | pm10_24h | 可吸入颗粒物24小时均值 |
| 3 | no2_24h | 二氧化氮24小时均值 |
| 4 | so2_24h | 二氧化硫24小时均值 |
| 5 | co_24h | 一氧化碳24小时均值 |
| 6 | o3_8h_24h | 臭氧8小时滑动均值 |
| 7 | month | 月份（1-12） |
| 8 | season | 季节（1-4） |

**输出目标（7维）**：前7项污染物浓度预测值

---

## 2. 模型架构详解

### 2.1 Transformer

**架构原理**

Transformer基于自注意力机制（Self-Attention Mechanism），能够并行处理序列中所有时间步，捕捉任意距离的依赖关系。

**网络结构**：

```
输入层 (14×9)
    ↓
线性投影层: 9维 → 128维 (d_model)
    ↓
位置编码层: 注入序列位置信息 (sinusoidal encoding)
    ↓
Transformer编码器 × 4层
  ├── 多头自注意力层 (nhead=4)
  ├── 前馈神经网络 (d_ff=512)
  └── 残差连接 + 层归一化
    ↓
解码层: 128维 → 7维
    ↓
输出层 (14×7)
```

**核心机制**：

自注意力计算公式：
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

其中Q（Query）、K（Key）、V（Value）均由输入序列线性变换得到。该机制允许模型在每个时间步关注序列中所有其他时间步，从而捕捉长期依赖关系。

**代码实现**（`src/air_quality/models/transformer.py`）：
```python
self.encoder = nn.Linear(input_size, d_model)        # 输入投影
self.pos_encoder = PositionalEncoding(d_model)        # 位置编码
self.transformer_encoder = nn.TransformerEncoder(     # 4层编码器
    nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512),
    num_layers=4
)
self.decoder = nn.Linear(d_model, output_size)        # 输出投影
```

**优势**：
- 并行计算效率高
- 能捕捉任意距离的时序依赖
- 通过多头注意力学习多尺度特征

**局限**：
- 对位置信息的建模依赖显式位置编码
- 计算复杂度为O(n²)，对长序列计算开销较大

---

### 2.2 LSTM

**架构原理**

长短期记忆网络（Long Short-Term Memory）通过门控机制解决传统RNN的梯度消失问题，能够学习长期依赖关系。

**网络结构**：

```
输入层 (14×9)
    ↓
双向LSTM层 × 4层
  ├── 前向LSTM: 按时间顺序处理
  ├── 后向LSTM: 按时间逆序处理
  └── 每层隐藏单元数: 256
    ↓
全连接层1: 512维 → 256维 (ReLU激活)
    ↓
Dropout层 (p=0.05)
    ↓
全连接层2: 256维 → 7维
    ↓
输出层 (14×7)
```

**门控机制**：

LSTM单元包含三个门控结构：

1. **遗忘门（Forget Gate）**：
   ```
   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
   ```
   决定从细胞状态中丢弃哪些信息。

2. **输入门（Input Gate）**：
   ```
   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
   C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
   ```
   决定将哪些新信息存入细胞状态。

3. **输出门（Output Gate）**：
   ```
   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
   h_t = o_t * tanh(C_t)
   ```
   决定从细胞状态中输出哪些信息。

**代码实现**（`src/air_quality/models/lstm.py`）：
```python
self.lstm = nn.LSTM(
    input_size,           # 输入维度: 9
    hidden_size=256,      # 隐藏状态维度: 256
    num_layers=4,         # LSTM层数: 4
    dropout=0.05,         # 层间dropout
    batch_first=True,     # 输入格式: (batch, seq, feature)
    bidirectional=True    # 双向LSTM
)
```

**优势**：
- 专为序列数据设计，天然适合时序预测
- 门控机制有效缓解梯度消失问题
- 双向结构可同时利用过去和未来信息

**局限**：
- 顺序计算，无法并行化
- 对极长序列仍可能存在信息瓶颈

---

### 2.3 CNN

**架构原理**

卷积神经网络通过局部感受野提取时序数据的局部特征模式，采用多尺度卷积核捕获不同时间尺度的模式。

**网络结构**：

```
输入层 (14×9)
    ↓
维度转换: (batch, 14, 9) → (batch, 9, 14)
    ↓
多尺度卷积（并行）
  ├── Conv1d(kernel=3): 捕获3时间步局部模式
  ├── Conv1d(kernel=5): 捕获5时间步局部模式
  └── Conv1d(kernel=7): 捕获7时间步局部模式
  每个分支包含: Conv1d → BatchNorm → ReLU → MaxPool1d
    ↓
特征拼接: 64×3 = 192维
    ↓
自适应平均池化: 保持序列长度不变
    ↓
维度转换: (batch, 192, 14) → (batch, 14, 192)
    ↓
全连接层1: 192维 → 128维 (ReLU激活)
    ↓
Dropout层
    ↓
全连接层2: 128维 → 7维
    ↓
输出层 (14×7)
```

**多尺度卷积核设计原理**：

- **kernel_size=3**：捕捉短期趋势（如连续3天的污染物变化）
- **kernel_size=5**：捕捉中期模式（如工作日/周末差异）
- **kernel_size=7**：捕捉周级别周期性

**代码实现**（`src/air_quality/models/cnn.py`）：
```python
self.convs = nn.ModuleList()
for kernel_size in [3, 5, 7]:
    self.convs.append(nn.Sequential(
        nn.Conv1d(in_channels=input_size, out_channels=64,
                  kernel_size=kernel_size, padding=kernel_size // 2),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.MaxPool1d(2, stride=1, padding=1)
    ))
```

**优势**：
- 局部特征提取能力强
- 参数共享减少模型参数量
- 多尺度设计可同时捕获不同时间粒度的模式

**局限**：
- 感受野有限，难以捕捉超长期依赖
- 对序列顺序信息的建模能力较弱

---

### 2.4 Hybrid

**架构原理**

混合模型通过集成Transformer、LSTM、CNN三个分支，并使用交叉注意力机制进行特征融合，综合利用不同架构的优势。

**网络结构**：

```
输入层 (14×9)
    ↓
输入投影层: 9维 → 128维
    ↓
┌───────────────────────────────────────────────────┐
│               三路并行编码分支                       │
├───────────────────┬───────────────┬───────────────┤
│  Transformer分支   │ LSTM分支      │ CNN分支        │
├───────────────────┼───────────────┼───────────────┤
│ PositionalEncoding│ 双向LSTM×2层   │ 多尺度Conv1d   │
│ TransformerEnc×2  │ FC(256→128)   │ (kernel 3,5,7)│
│                   │ ReLU          │ BN+ReLU+Pool  │
│                   │               │ FC(192→128)   │
└───────────────────┴───────────────┴───────────────┘
         ↓                 ↓               ↓
┌───────────────────────────────────────────────────┐
│               轻量级自注意力层                       │
├───────────────────┬───────────────┬───────────────┤
│ T_self_attn       │ L_self_attn   │ C_self_attn   │
│ (nhead=2)         │ (nhead=2)     │ (nhead=2)     │
└───────────────────┴───────────────┴───────────────┘
         ↓                 ↓               ↓
┌───────────────────────────────────────────────────┐
│           交叉注意力融合层 (CrossAttentionFusion)    │
├───────────────────────────────────────────────────┤
│ 6路交叉注意力:                                      │
│   T↔L (cross_attn_tl, cross_attn_lt)              │
│   T↔C (cross_attn_tc, cross_attn_ct)              │
│   L↔C (cross_attn_lc, cross_attn_cl)              │
│                                                   │
│ 可学习融合权重: weights = softmax(w_t, w_l, w_c)     │
│ 融合公式: fused = w_t×T_fused + w_l×L_fused + w_c×C_fused │
└───────────────────────────────────────────────────┘
         ↓
残差连接: output_fc(fused + 0.3 × identity)
         ↓
输出层: FC(128→64) → ReLU → Dropout → FC(64→7)
         ↓
输出层 (14×7)
```

**交叉注意力机制**：

以Transformer分支从LSTM分支获取信息为例：
```python
# Query来自Transformer，Key/Value来自LSTM
t_from_l, _ = cross_attn_tl(t_proj, l_proj, l_proj)
# 残差连接 + 层归一化
t_fused = layer_norm(t_proj + t_from_l + t_from_c)
```

该机制允许各分支相互"学习"其他分支的特征表示，实现信息互补。

**代码实现**（`src/air_quality/models/hybrid.py`）：
```python
class HybridModel(BaseModel):
    def __init__(self, ...):
        self.input_proj = nn.Linear(input_size, d_model)
        self.transformer_branch = TransformerBranch(d_model, nhead, num_layers=2)
        self.lstm_branch = LSTMEncoderBranch(d_model, d_model, num_layers=2)
        self.cnn_branch = CNNEncoderBranch(d_model, num_filters=64)
        self.fusion = CrossAttentionFusion(d_model, nhead)
        self.residual_weight = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        identity = self.input_proj(x)
        t_out = self.transformer_branch(identity)
        l_out = self.lstm_branch(identity)
        c_out = self.cnn_branch(identity)
        # 自注意力精炼
        t_out = self.t_self_attn(t_out)
        l_out = self.l_self_attn(l_out)
        c_out = self.c_self_attn(c_out)
        # 交叉注意力融合
        fused = self.fusion(t_out, l_out, c_out)
        return self.output_fc(fused + self.residual_weight * identity)
```

**优势**：
- 集成多种架构优势，特征提取更全面
- 交叉注意力实现分支间信息交互
- 可学习的融合权重自适应调整各分支贡献

**局限**：
- 模型参数量较大
- 训练时间较长
- 可能存在过拟合风险

---

## 3. 监督学习适配性分析

本项目采用标准的监督学习范式，所有模型均适配该学习框架。

**监督学习定义**：给定输入-输出对 (X, y)，学习映射函数 f: X → y

**本项目的监督学习设置**：
- 输入X：过去14天的9维特征序列，形状 (14, 9)
- 输出y：未来3天的7维预测值，形状 (3, 7)
- 损失函数：Huber Loss（默认）/ MSE / MAE
- 优化器：Adam + OneCycleLR学习率调度

**各模型适配性评估**：

| 模型 | 适配度 | 评估依据 |
|------|--------|----------|
| LSTM | ★★★★★ | 专为序列建模设计，门控机制天然适合监督学习 |
| Transformer | ★★★★☆ | 自注意力机制强大，但需要较多数据支撑 |
| CNN | ★★★★☆ | 局部特征提取能力强，适合短期模式识别 |
| Hybrid | ★★★★★ | 集成多架构优势，理论性能上限最高 |

**损失函数选择**：

本项目默认使用Huber Loss，该损失函数结合了MSE和MAE的优点：
```
L_δ(a) = {
  0.5 × a²           if |a| ≤ δ
  δ × (|a| - 0.5δ)   otherwise
}
```
- 当误差较小时（|a| ≤ δ），表现为MSE，梯度平滑
- 当误差较大时，表现为MAE，对异常值鲁棒

---

## 4. 准确性提升方案

### 4.1 推荐新增算法

#### 4.1.1 XGBoost / LightGBM

**适用理由**：
- 在结构化/表格数据上表现优异，常作为Kaggle竞赛的强基线模型
- 训练速度快，无需GPU
- 内置特征重要性评估
- 对缺失值和异常值具有较强鲁棒性

**实现方案**：
```python
# 将时序数据展平为表格形式
# 输入: (14天 × 9特征) = 126个特征
X_flat = X.reshape(X.shape[0], -1)

# 使用XGBoost进行多输出回归
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

model = MultiOutputRegressor(XGBRegressor(n_estimators=1000))
model.fit(X_flat, y.reshape(y.shape[0], -1))
```

#### 4.1.2 Prophet

**适用理由**：
- Facebook开源的时序预测库，专为业务时序数据设计
- 自动检测并建模多种季节性（周、月、年）
- 对节假日效应有显式建模
- 对缺失数据和异常值鲁棒性强

**实现方案**：
```python
from prophet import Prophet

# 对每个污染物单独建模
for pollutant in ['aqi', 'pm2_5', ...]:
    df_prophet = df[['date', pollutant]].rename(columns={'date': 'ds', pollutant: 'y'})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df_prophet)
```

#### 4.1.3 WaveNet

**适用理由**：
- Google DeepMind提出的生成模型，采用因果膨胀卷积
- 相比TCN实现更成熟，无维度匹配bug
- 在语音生成任务中验证了其对时序数据的强大建模能力
- 感受野可指数级增长

**核心结构**：
```
因果膨胀卷积堆叠:
  Layer 1: dilation=1, kernel=2
  Layer 2: dilation=2, kernel=2
  Layer 3: dilation=4, kernel=2
  ...
  Layer n: dilation=2^(n-1), kernel=2

总感受野: 2^n
```

#### 4.1.4 Informer

**适用理由**：
- 专为长序列时序预测设计（Long Sequence Time-Series Forecasting）
- 提出ProbSparse注意力机制，计算复杂度从O(n²)降至O(n·log n)
- 在ETT（Electricity Transformer Temperature）数据集上达到SOTA
- 自蒸馏机制减少特征冗余

#### 4.1.5 N-BEATS

**适用理由**：
- 纯神经网络时序预测架构，无需特征工程
- 可解释性强，通过趋势和季节性分解提供预测依据
- 在M4竞赛中获得冠军
- 模块化设计便于扩展

### 4.2 特征工程优化

当前系统仅使用9个基础特征，建议从以下维度扩展：

#### 4.2.1 气象特征

```python
meteorological_features = {
    'temperature': '温度',
    'humidity': '相对湿度',
    'wind_speed': '风速',
    'wind_direction': '风向',
    'pressure': '气压',
    'precipitation': '降水量',
    'visibility': '能见度'
}
```

**理论依据**：气象条件直接影响污染物扩散和转化，是空气质量的重要影响因子。

#### 4.2.2 时间特征增强

```python
temporal_features = {
    'day_of_week': '星期几（0-6）',
    'is_weekend': '是否周末（0/1）',
    'is_holiday': '是否法定假日（0/1）',
    'day_of_year': '年内天数（1-366）',
    'week_of_year': '年内周数（1-53）',
    'is_heating_season': '是否供暖季（0/1）'
}
```

**理论依据**：人类活动具有明显的时间周期性，供暖季燃煤排放是冬季污染的重要来源。

#### 4.2.3 滞后特征

```python
lag_features = {
    'aqi_lag_1': '前一天AQI',
    'aqi_lag_7': '一周前AQI',
    'aqi_lag_30': '一月前AQI',
    'aqi_rolling_7d_mean': '近7天AQI均值',
    'aqi_rolling_7d_std': '近7天AQI标准差',
    'aqi_diff_1': 'AQI日变化量',
    'aqi_diff_7': 'AQI周变化量'
}
```

**理论依据**：时序数据具有自相关性，历史值是未来值的重要预测因子。

#### 4.2.4 地理特征

```python
geographical_features = {
    'station_id': '监测站点ID',
    'latitude': '纬度',
    'longitude': '经度',
    'elevation': '海拔',
    'urbanization_level': '城市化水平'
}
```

### 4.3 模型融合策略

#### 4.3.1 加权平均融合

```python
# 基于验证集性能的加权融合
def weighted_ensemble(predictions: dict, weights: dict) -> np.ndarray:
    """
    predictions: {'model_name': pred_array}
    weights: {'model_name': weight}
    """
    result = sum(predictions[name] * weights[name] for name in predictions)
    return result

# 示例
weights = {
    'transformer': 0.3,
    'lstm': 0.25,
    'xgboost': 0.25,
    'prophet': 0.2
}
```

#### 4.3.2 Stacking集成

```python
# 使用元学习器学习最优融合权重
from sklearn.linear_model import RidgeCV

# 第一层：多个基模型的预测结果
base_predictions = np.column_stack([
    transformer_pred, lstm_pred, xgboost_pred, prophet_pred
])

# 第二层：元学习器
meta_learner = RidgeCV(alphas=[0.1, 1.0, 10.0])
meta_learner.fit(base_predictions, y_true)
final_pred = meta_learner.predict(base_predictions)
```

#### 4.3.3 Bagging集成

```python
# 对同一模型的多次训练结果进行集成
from sklearn.utils import resample

ensemble_preds = []
for i in range(n_models):
    # 不同的随机种子或数据子集
    X_boot, y_boot = resample(X_train, y_train, random_state=i)
    model_i = train_model(X_boot, y_boot)
    ensemble_preds.append(model_i.predict(X_test))

final_pred = np.mean(ensemble_preds, axis=0)
```

### 4.4 架构改进方向

#### 4.4.1 Attention-LSTM

为LSTM添加注意力机制，使模型能够自适应地关注重要的时间步：

```python
class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context)
```

#### 4.4.2 Transformer Encoder-Decoder架构

当前Transformer仅使用Encoder，改用Encoder-Decoder架构更适合序列到序列预测任务：

```python
class TransformerED(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_encoder_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead), num_decoder_layers
        )

    def forward(self, src, tgt):
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return output
```

#### 4.4.3 修复TCN实现

```python
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=self.padding, dilation=dilation)
        # ... 其他层

    def forward(self, x):
        out = self.conv1(x)
        # 裁剪输出以匹配输入长度
        out = out[:, :, :x.size(2)]
        # 残差连接
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
```

---

## 5. 综合评估与建议

### 5.1 当前系统评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 架构设计 | ★★★★☆ | Hybrid模型融合思路正确，但可选模型较少 |
| 特征工程 | ★★☆☆☆ | 仅使用9个基础特征，未充分利用外部信息 |
| 模型多样性 | ★★★★☆ | 4个模型，功能互补 |
| 可解释性 | ★★☆☆☆ | 缺乏特征重要性分析和预测归因 |
| 工程实现 | ★★★★☆ | 代码结构清晰，模块化程度高 |

### 5.2 优先级排序

| 优先级 | 改进项 | 预期收益 |
|--------|--------|----------|
| P0 | 添加气象特征 | 预计提升5-15% |
| P0 | 添加XGBoost基线模型 | 提供强基线参照 |
| P1 | 实现多模型融合 | 预计提升3-8% |
| P1 | 添加WaveNet模型 | 增加模型多样性 |
| P2 | 添加滞后特征 | 预计提升2-5% |
| P2 | 实现Informer/N-BEATS | 探索更优架构 |
| P3 | 添加SHAP可解释性分析 | 提升模型可信度 |

### 5.3 预期效果

通过实施上述改进方案，预计系统预测准确性可提升10-20%，具体取决于：
- 外部数据的获取难度和质量
- 模型融合策略的选择
- 超参数调优的充分程度

---

## 6. 数据预处理与可视化一致性

### 6.1 异常值裁剪的两面性

`AirQualityDataProcessor._handle_outliers` 在数据加载阶段对污染物做 IQR 异常值裁剪：

| 特征 | 上限策略 | 物理含义 |
|------|----------|----------|
| AQI | 硬编码 ≤ 500 | 中国 AQI 最大值 |
| CO | 硬编码 ≤ 50 | 单位 mg/m³ 实际量级 |
| PM2.5 / PM10 / NO2 / SO2 / O3 | `Q3 + 3×IQR` 统计上限 | 抑制传感器异常 |

**训练侧的好处**：避免极端值主导损失函数，让 Adam 优化更稳定。
**可视化的副作用**：在回测图表中，"真实值"曲线被裁剪后的范围强行限制，matplotlib 自动按裁剪范围决定 y 轴——这会让极端污染事件（AQI > 500、CO > 50、PM2.5 > 500）在图中"消失"或被压扁成平直线。

### 6.2 Y 轴准确性修复

修复方案：`plot_backtest_results` 接受可选参数 `y_true_raw`，提供时优先使用原始未裁剪真值，未提供时回退到 `y_true`（向后兼容）。

```python
# scripts/evaluate.py
import pandas as pd
raw_df = pd.read_excel(args.data_file, engine='openpyxl')
column_mapping = processor._create_column_mapping(raw_df.columns)
raw_df = raw_df.rename(columns=column_mapping)
raw_pollutants = raw_df[features].values
# 按预测窗口对齐 (start + seq_len, start + seq_len + prediction_days)
y_true_raw = np.array([
    raw_pollutants[start + seq_len:start + seq_len + args.days]
    for start in range(n_samples)
])

plot_backtest_results(
    y_true=result['y_true'],   # 用于指标计算（被裁剪过）
    y_pred=result['y_pred'],
    dates=result['dates'],
    y_true_raw=y_true_raw,     # 用于绘图，y 轴反映真实数据范围
)
```

### 6.3 设计原则

- **训练与可视化解耦**：训练需要异常值裁剪提升鲁棒性；可视化需要原始数据反映真实分布
- **不修改训练行为**：修复只影响图表绘制，不改变模型输入/输出或损失函数
- **优雅降级**：当原始数据列数不足时自动回退到旧行为
- **回归测试覆盖**：新增 `test_y_true_raw_overrides_y_true` 和 `test_backward_compatible_without_y_true_raw` 保证行为

### 6.4 适用范围

- **回测图表**：`scripts/evaluate.py` 自动启用 `y_true_raw`
- **其他绘图函数**：`plot_training_history` / `plot_prediction_comparison` / `plot_feature_comparison` / `plot_metrics` 均为训练过程可视化，无需此修复
- **生产预测**：单点预测不影响 y 轴，无需处理

---

## 参考文献

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." Neural Computation.
3. Bai, S., Kolter, J. Z., & Koltun, V. (2018). "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling." arXiv.
4. Zhou, H., et al. (2021). "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting." AAAI.
5. Oreshkin, B., et al. (2020). "N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting." ICLR.
6. van den Oord, A., et al. (2016). "WaveNet: A Generative Model for Raw Audio." arXiv.

---

*文档生成日期：2026-06-16*
*项目版本：v0.0.2*
*最近更新：补充数据预处理与可视化一致性章节（`y_true_raw` 修复回测图表 Y 轴截断问题）*
