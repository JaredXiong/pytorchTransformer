# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

空气质量时间序列预测系统，基于 PyTorch。基于 14 天输入序列（9 维特征），预测未来 3 天的 AQI 及污染物浓度（PM2.5、PM10、NO2、SO2、CO、O3）。数据集：北京 2015-2024 空气质量数据，存放在 `data/raw/北京2015-2024.xlsx`。

## 语言要求

请始终使用简体中文输出

## 目录结构

```
pytorchTransformer/
├── CLAUDE.md                       # 本文件
├── README.md                       # 项目说明文档
├── requirements.txt                # 运行时依赖
├── .gitignore
├── docs/
│   ├── algorithm_analysis.md       # 算法架构分析文档
│   └── superpowers/                # 设计历史（specs + plans，详见末尾「设计文档」）
│       ├── specs/
│       └── plans/
├── data/
│   └── raw/                        # 原始数据集
├── outputs/                        # 训练产物（git 忽略）
│   ├── checkpoints/                # 模型权重与标准化器
│   │   ├── air_quality_model.pth
│   │   └── scaler.pkl
│   └── figures/                    # 可视化图像
├── src/
│   └── air_quality/                # 核心 Python 包
│       ├── __init__.py             # 公开 API
│       ├── config/                 # dataclass 集中配置
│       ├── data/                   # 数据加载、预处理、序列构造
│       ├── models/                 # 模型架构
│       │   ├── base.py             # BaseModel + PositionalEncoding
│       │   ├── transformer.py      # TransformerModel
│       │   ├── lstm.py             # LSTMModel
│       │   ├── cnn.py              # CNNModel
│       │   ├── hybrid.py           # HybridModel + Cross-Attention Fusion
│       │   ├── ensemble.py         # EnsembleModel（加权平均融合多模型）
│       │   └── factory.py          # create_model()
│       ├── training/               # 训练器、损失、指标
│       │   ├── trainer.py
│       │   ├── losses.py
│       │   └── metrics.py
│       ├── inference/              # 加载权重、自回归预测
│       │   └── predictor.py
│       └── visualization/          # 训练曲线与预测对比图
│           └── plots.py
├── scripts/                        # 命令行入口
│   ├── train.py                    # 统一训练脚本（支持默认、调优、深度调优、集成学习）
│   ├── compare_models.py           # 多模型对比
│   └── evaluate.py                 # 回测评估 + 7 子图可视化
└── tests/                          # 单元测试（unittest 框架，详见「测试」）
    ├── test_config.py
    ├── test_models.py
    ├── test_processor.py
    ├── test_metrics.py
    ├── test_predictor.py
    └── test_visualization.py
```

## 运行方式

```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) 训练默认的 Hybrid 模型并产出 outputs/checkpoints/ 下的权重
python -m scripts.train

# 3) 训练其他模型类型
python -m scripts.train --model lstm
python -m scripts.train --model transformer

# 4) 对比所有模型
python -m scripts.compare_models
python -m scripts.compare_models --models transformer lstm cnn hybrid

# 5) 回测评估：训练后用全量数据做滚动预测，画 7 子图真值 vs 预测对比
python -m scripts.evaluate
python -m scripts.evaluate --model lstm --days 3

# 6) 以库方式使用
python -c "from air_quality import config, create_model, ModelTrainer, AirQualityPredictor; print(config.model.input_size)"
```

## 测试

测试使用 stdlib `unittest`（未引入 pytest），位于 `tests/`。

```bash
# 跑全部测试（18 个测试，耗时约 13 秒）
python -m unittest discover tests -v

# 跑单个测试文件
python -m unittest tests.test_config -v
python -m unittest tests.test_models -v
python -m unittest tests.test_predictor -v

# 跑单个测试方法
python -m unittest tests.test_config.TestCalendarConfig.test_input_size_unchanged -v
python -m unittest tests.test_predictor.TestForecast.test_forecast_rejects_invalid_month -v
```

## 架构设计

### 配置（`src/air_quality/config/settings.py`）

全局单例 `config` 聚合 6 个 dataclass 子配置。关键默认值：

| 项 | 值 |
|---|---|
| `ModelConfig.input_size` | 9（7 污染物 + month + season）|
| `ModelConfig.output_size` | 7（仅污染物，不含 month/season）|
| `ModelConfig.d_model` | 128 |
| `ModelConfig.lstm_hidden` | 256 |
| `ModelConfig.cnn_filters` | 128 |
| `ModelConfig.nhead / num_layers` | 4 |
| `TrainingConfig.epochs` | 250 |
| `TrainingConfig.batch_size` | 32 |
| `TrainingConfig.learning_rate` | 0.0003 |
| `TrainingConfig.weight_decay` | 3e-4 |
| `TrainingConfig.early_stop_patience` | 60 |
| `TrainingConfig.loss_type` | `huber` |
| `TrainingConfig.gradient_clip` | 1.0 |
| `DataConfig.data_file` | `data/raw/北京2015-2024.xlsx` |
| `DataConfig.seq_length` | 14 |
| `DataConfig.prediction_days` | 3 |
| `DataConfig.train_split_ratio` | 0.8 |
| `DataConfig.feature_columns` | 9 个：7 污染物 + month + season（输入）|
| `DataConfig.target_columns` | 7 个污染物（输出，不含 month/season）|
| `DataConfig.rolling_windows` | `[]`（已禁用以避免维度不匹配）|
| `FileConfig.model_save_path` | `outputs/checkpoints/air_quality_model.pth` |
| `FileConfig.scaler_path` | `outputs/checkpoints/scaler.pkl` |
| `FileConfig.figures_dir` | `outputs/figures` |

通过 `config.update_from_dict({...})` 运行时覆盖任意字段。

### 数据处理（`src/air_quality/data/processor.py`）

- `AirQualityDataProcessor.load_and_preprocess_data`：读 Excel → 列名映射（中英文 → `_24h` 标准名）→ IQR 异常值裁剪（AQI≤500、CO≤50、其他 3×IQR）→ `SimpleImputer(mean)` → 可选 `rolling_windows` 滚动统计（仅作用于污染物）→ `StandardScaler`（**仅拟合 7 个污染物**）→ 拼接原始 `month` / `season` 整数。返回 `data` 形状 `(N, 9)`：前 7 列缩放污染物 + 后 2 列原始 calendar。
- `AirQualityDataProcessor.create_sequences`：滑窗构造 (X, y) 序列对，**X 形状 `(N, 14, 9)`**（含 month/season），**y 形状 `(N, 3, 7)`**（仅污染物）。
- `validate_prediction`：物理边界裁剪（仅作用于 7 维输出：AQI 索引 0 ≤ 500，CO 索引 5 ≤ 50；不再验证 month/season）。
- `calculate_metrics`：per-feature + overall MSE/RMSE/MAE/R²，默认 7 个污染物。
- `get_device`：自动选择 CUDA → MPS → CPU。

### 模型（`src/air_quality/models/`）

`BaseModel` 提供统一权重初始化与 `count_parameters`。共 4 种模型，全部通过 `create_model(model_type, input_size, output_size=7, **kwargs)` 工厂实例化：

| `model_type` | 类 | 关键结构 |
|---|---|---|
| `transformer` | `TransformerModel` | Linear 投影 → PositionalEncoding → 4 层 TransformerEncoder (d_ff=512) → Linear 解码 |
| `lstm` | `LSTMModel` | 4 层双向 LSTM (hidden=256, dropout=0.05) → 2× Linear |
| `cnn` | `CNNModel` | 多尺度 Conv1d（kernel [3,5,7]）+ BN+ReLU+MaxPool → AdaptiveAvgPool → 2× Linear |
| `hybrid` | `HybridModel` | 共享 `input_proj` → TransformerBranch + LSTMEncoderBranch + CNNEncoderBranch → 各 `LightSelfAttention` → `CrossAttentionFusion`（6 路交叉注意力 + 可学习融合权重）→ FC + 可学习残差 |

**新增模型**：在 `src/air_quality/models/` 下新建 `<name>.py`（继承 `BaseModel`），在 `factory.py` 的 `model_classes` 注册键。所有模型 forward 必须返回与输入等长的序列（time-step 数不变），最终层维度为 `output_size`（默认 7 = 仅污染物）。

**集成模型**：`EnsembleModel` 通过加权平均融合多个模型的预测，用于提升预测稳定性。

### 训练（`src/air_quality/training/`）

- `ModelTrainer`：Adam + 可配置 loss（`huber` / `mse` / `mae` / `smooth_l1`）+ 梯度裁剪 + OneCycleLR 调度器 + 早停。`_train_epoch` / `_validate_epoch` 取 `output[:, -prediction_days:, :]` 对齐 target（所有模型 forward 必须返回与输入等长的序列）。
- `MetricsTracker`：per-feature + overall MSE/RMSE/MAE/R²，含 `detect_over_smoothing`。
- `get_loss_function` / `WeightedMSELoss`：工厂与加权损失。

### 推理（`src/air_quality/inference/predictor.py`）

两种推理模式：

- **生产预测 `AirQualityPredictor.forecast(input_sequence, future_dates=None)`**
  - 输入：原始尺度 numpy 数组 `(14, 9)`，前 7 列污染物 + 后 2 列 month/season
  - 流程：缩放仅作用于前 7 列污染物 → 自回归滚动 `prediction_days` 次（每步保留上一时间步的 month/season 拼回）→ 逆缩放
  - 输入验证：每列取值范围检查（`validate_input`），越界抛 `ValueError`
- **回测评估 `AirQualityPredictor.backtest(data, dates, prediction_days=3)`**
  - 输入：完整数据集 `(N, 9)`（前 7 缩放 + 后 2 原始，processor 输出）+ 日期 Series
  - 流程：对每个 14 天窗口滚动预测 3 天，与真实值对比
  - 输出：`y_true` / `y_pred` / `dates` / `metrics`，配合 `plot_backtest_results` 画 7 子图对比

加载权重时从检查点 `model_type` 自动推断模型类名。`predict_air_quality` 便捷函数封装 `forecast()`。

**forecast 输入契约**：

| 列 | 名称 | 范围 | 类型 |
|---|---|---|---|
| 0 | aqi | 0–500 | float |
| 1 | pm2_5_24h | 0–1000 | float |
| 2 | pm10_24h | 0–1000 | float |
| 3 | no2_24h | 0–500 | float |
| 4 | so2_24h | 0–500 | float |
| 5 | co_24h | 0–50 | float |
| 6 | o3_8h_24h | 0–500 | float |
| 7 | month | 1–12 | int |
| 8 | season | 1–4 | int |

`PredictionResult.AQI_GRADES`：六级分类（优/良/轻度/中度/重度/严重污染）；`format_prediction_result` 输出可读表格。

### 可视化（`src/air_quality/visualization/plots.py`）

5 个函数：`plot_training_history` / `plot_prediction_comparison` / `plot_feature_comparison`（含误差分布）/ `plot_metrics` / **`plot_backtest_results`**（每个污染物独立一张图，按模型保存到 `outputs/figures/<model_type>/`）。matplotlib 后端强制 `Agg` 非交互式，支持中文（`SimHei` / `Microsoft YaHei`）。

**`plot_backtest_results` 关键参数**：

| 参数 | 类型 | 含义 |
|---|---|---|
| `y_true` | `(N, prediction_days, 7)` | 模型推理时获取的真值（来自预处理后的数据，已被 `_handle_outliers` 裁剪）|
| `y_pred` | `(N, prediction_days, 7)` | 模型预测值 |
| `y_true_raw` | `(N, prediction_days, 7)` 可选 | **原始未裁剪真值**。若传入，图表使用它绘制"真实值"曲线，让 y 轴反映真实数据范围 |

**Y 轴准确性修复**：`_handle_outliers` 在预处理时将 AQI 硬编码上限 500、CO 上限 50、其他污染物用 `Q3+3IQR`，导致 `y_true` 被强行截断、matplotlib 自动按截断后的范围决定 y 轴。`scripts/evaluate.py` 会重新加载 Excel 原始数据并按窗口对齐成 `y_true_raw` 传入图表函数，确保极端污染事件（AQI > 500、CO > 50、PM2.5/PM10 > 500 等）能正常显示。

## 数据特征（9 维）

按 `AirQualityDataProcessor.feature_columns` 顺序：`aqi, pm2_5_24h, pm10_24h, no2_24h, so2_24h, co_24h, o3_8h_24h, month, season`。`month` / `season` 由 `pubtime` / `date` 列派生。Excel 至少需要日期列与 ≥3 个污染物列；列名匹配支持中文别名（细颗粒物、可吸入颗粒物、二氧化氮、二氧化硫、一氧化碳、臭氧、空气质量指数）。

## 常见改动指引

- **调参**：编辑 `src/air_quality/config/settings.py`，或在运行时用 `config.update_from_dict(...)`。
- **新增模型**：在 `src/air_quality/models/<name>.py` 实现类，编辑 `factory.py` 注册。
- **新增损失/指标**：在 `src/air_quality/training/losses.py` 或 `metrics.py` 添加，在 `__init__.py` 导出。
- **换数据文件**：把 Excel 放到 `data/raw/` 下，或修改 `DataConfig.data_file`。
- **改预测天数**：`DataConfig.prediction_days`；scaler 训练时的特征数（9）需 ≥ 该值。
- **改训练入口参数**：`scripts/train.py` 已支持 `--model` / `--epochs` / `--batch-size` / `--lr`，可继续扩展。

## 设计文档

架构决策与变更历史在 `docs/superpowers/` 下保留：

- `docs/superpowers/specs/` — 设计 spec（如 `2026-06-15-calendar-features-design.md`）
- `docs/superpowers/plans/` — 实施 plan（如 `2026-06-15-calendar-features.md`）

修改或扩展架构前先读相关 spec 了解历史背景。

## Python 环境

需要 Python 3.9+，建议使用项目专用的 conda 环境（已存在）：

```bash
conda activate pytorchTransformer
```

PyTorch、scikit-learn、matplotlib 等核心依赖见 `requirements.txt`。
