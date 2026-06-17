# 空气质量时间序列预测系统

基于 PyTorch 的空气质量时间序列预测系统，使用深度学习模型预测未来 3 天的 AQI 及污染物浓度。

## 项目概述

本项目实现了一个完整的空气质量预测系统，基于北京 2015-2024 年的历史空气质量数据，通过多种深度学习模型进行时序预测。

**核心功能**：
- 输入：过去 14 天的 9 维特征（7 个污染物 + 月份 + 季节）
- 输出：未来 3 天的 7 个污染物浓度预测（AQI、PM2.5、PM10、NO2、SO2、CO、O3）

## 支持的模型

| 模型 | 类型 | 特点 |
|------|------|------|
| Transformer | 自注意力机制 | 全局依赖建模，并行计算效率高 |
| LSTM | 门控循环网络 | 长期依赖建模，序列数据天然适配 |
| CNN | 多尺度卷积 | 局部特征提取，训练速度最快 |
| Hybrid | 混合模型 | 集成 Transformer + LSTM + CNN，交叉注意力融合 |
| VMD-CNN-BiLSTM-Attention | VMD + 多尺度 CNN + BiLSTM + 注意力 | 专为空气质量预测设计，支持半监督与预训练-微调 |
| Ensemble | 集成模型 | 加权平均融合多个基模型，提升预测稳定性 |

## 快速开始

### 环境要求

- Python 3.9+
- PyTorch 1.8+
- 其他依赖见 `requirements.txt`

### 安装

```bash
# 克隆项目
git clone <repository-url>
cd pytorchTransformer

# 创建 conda 环境（推荐）
conda create -n pytorchTransformer python=3.9
conda activate pytorchTransformer

# 安装依赖
pip install -r requirements.txt
```

### 训练模型

```bash
# 训练默认 Hybrid 模型
python -m scripts.train

# 训练其他模型
python -m scripts.train --model lstm
python -m scripts.train --model transformer
python -m scripts.train --model cnn

# 自定义超参数
python -m scripts.train --model lstm --epochs 50 --batch-size 64 --lr 0.0005

# 对比所有模型
python -m scripts.compare_models

# 对比指定模型
python -m scripts.compare_models --models transformer lstm cnn hybrid
```

### 半监督训练 (VMD-CNN-BiLSTM-Attention)

数据划分：40% 有标签 + 40% 无标签（伪标签扩展）+ 20% 测试。

```bash
# 启用半监督（默认 VMD K=4）
python -m scripts.train --semi-supervised

# 自定义 VMD 模态数
python -m scripts.train --semi-supervised --vmd-k 6

# 关闭 VMD，对比纯半监督
python -m scripts.train --semi-supervised --no-vmd

# 调节伪标签置信度阈值
python -m scripts.train --semi-supervised --pseudo-threshold 0.9

# 调节 Teacher / Student 训练轮数
python -m scripts.train --semi-supervised --teacher-epochs 100 --student-epochs 150
```

**注意**：启用 `--semi-supervised` 时会跳过默认的全监督训练（避免双倍时间），仅运行 VMD-CNN-BiLSTM-Attention 的 4 阶段半监督流程。

### 半监督 CLI 参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--semi-supervised` | 启用半监督训练 | 关闭 |
| `--vmd-k` | VMD 模态数 K | 4 |
| `--no-vmd` | 关闭 VMD 分解 | 启用 |
| `--pseudo-threshold` | 伪标签置信度阈值 | 0.85 |
| `--teacher-epochs` | Teacher 训练轮数 | 80 |
| `--student-epochs` | Student 训练轮数 | 120 |

### 预训练-微调半监督训练 (VMD-CNN-BiLSTM-Attention)

```bash
# 启用预训练-微调（必须在 --semi-supervised 之后）
python -m scripts.train --semi-supervised --pretrain

# 自定义预训练轮数
python -m scripts.train --semi-supervised --pretrain --pretrain-epochs 30

# 自定义掩码比例
python -m scripts.train --semi-supervised --pretrain --pretrain-mask-ratio 0.5
```

三阶段流水线：预训练（unlabeled 段掩码自监督）→ 伪标签（Teacher 推理）→ 微调（Student 联合训练）。

### 评估与预测

```bash
# 回测评估：每个污染物一张独立图表，保存到 outputs/figures/<model_type>/
python -m scripts.evaluate
python -m scripts.evaluate --model lstm --days 3
```

`evaluate.py` 会重新加载 Excel 原始数据并按预测窗口对齐，作为图表的"真实值"曲线，确保 y 轴能反映数据的真实范围（不被 `_handle_outliers` 的 IQR 裁剪强行压缩）。

### 以库方式使用

```python
from air_quality import config, create_model, ModelTrainer, AirQualityPredictor

# 创建模型
model = create_model('lstm', input_size=9, output_size=7)

# 预测
predictor = AirQualityPredictor()
predictor.load_model()
result = predictor.forecast(input_sequence)
```

## 项目结构

```
pytorchTransformer/
├── data/raw/                       # 原始数据集
├── docs/
│   ├── algorithm_analysis.md       # 算法架构分析
│   ├── math_principles.tex         # 算法数学原理（LaTeX）
│   └── superpowers/                # 设计历史（specs + plans）
├── outputs/                        # 训练产物（git 忽略）
│   ├── checkpoints/                # 模型权重与标准化器
│   └── figures/                    # 可视化图像
├── scripts/                        # 命令行入口
│   ├── train.py                    # 统一训练脚本
│   ├── compare_models.py           # 多模型对比
│   └── evaluate.py                 # 回测评估 + 7 子图可视化
├── src/air_quality/                # 核心 Python 包
│   ├── config/                     # dataclass 集中配置
│   ├── data/                       # 数据加载、预处理、序列构造
│   │   ├── processor.py            # 数据处理器
│   │   ├── vmd.py                  # VMD 变分模态分解
│   │   └── vmd_features.py         # VMD 特征变换
│   ├── models/                     # 模型架构
│   │   ├── base.py                 # BaseModel + PositionalEncoding
│   │   ├── transformer.py
│   │   ├── lstm.py
│   │   ├── cnn.py
│   │   ├── hybrid.py               # HybridModel + Cross-Attention Fusion
│   │   ├── vmd_cnn_bilstm_attention.py  # VMD-CNN-BiLSTM-Attention
│   │   ├── ensemble.py             # EnsembleModel
│   │   └── factory.py              # create_model()
│   ├── training/                   # 训练器、损失、指标
│   │   ├── trainer.py              # 全监督训练器
│   │   ├── semi_supervised.py      # 半监督训练器（伪标签法）
│   │   ├── pretrain.py             # 掩码自监督预训练器
│   │   └── pretrain_finetune.py    # 预训练-微调三阶段编排器
│   ├── inference/                  # 加载权重、自回归预测
│   └── visualization/              # 训练曲线与预测对比图
└── tests/                          # 单元测试（95 个）
```

## 配置说明

主要配置项在 `src/air_quality/config/settings.py`（实际值以代码为准）：

```python
# 模型配置
ModelConfig.input_size = 9       # 输入特征数（7 污染物 + month + season）
ModelConfig.output_size = 7      # 输出特征数（仅污染物）
ModelConfig.d_model = 128        # Transformer / Hybrid 隐藏维度
ModelConfig.lstm_hidden = 256    # LSTM 隐藏维度
ModelConfig.cnn_filters = 128    # CNN 过滤器数量
ModelConfig.nhead = 4            # 注意力头数
ModelConfig.num_layers = 4       # 模型层数

# 训练配置
TrainingConfig.epochs = 250           # 训练轮数
TrainingConfig.batch_size = 32        # 批大小
TrainingConfig.learning_rate = 0.0003 # 学习率
TrainingConfig.weight_decay = 3e-4    # 权重衰减
TrainingConfig.early_stop_patience = 60  # 早停耐心值
TrainingConfig.loss_type = 'huber'    # 损失函数
TrainingConfig.gradient_clip = 1.0    # 梯度裁剪

# 数据配置
DataConfig.seq_length = 14       # 输入序列长度
DataConfig.prediction_days = 3   # 预测天数
DataConfig.train_split_ratio = 0.8  # 训练集比例

# VMD 配置
VMDConfig.enabled = True         # 启用 VMD 分解
VMDConfig.K = 4                  # IMF 模态数
VMDConfig.alpha = 2000           # 带宽约束

# 半监督配置
SemiSupervisedConfig.enabled = False  # 默认关闭
SemiSupervisedConfig.labeled_ratio = 0.4
SemiSupervisedConfig.unlabeled_ratio = 0.4
SemiSupervisedConfig.test_ratio = 0.2
SemiSupervisedConfig.pseudo_confidence_threshold = 0.85

# 预训练配置
PretrainConfig.enabled = False    # 默认关闭
PretrainConfig.epochs = 60        # 预训练轮数
PretrainConfig.mask_ratio = 0.3   # 掩码比例
```

运行时覆盖配置：

```python
config.update_from_dict({
    'training': {'learning_rate': 0.0005, 'epochs': 200},
    'model': {'dropout': 0.3}
})
```

## 测试

测试使用 stdlib `unittest`（未引入 pytest），位于 `tests/`，共 95 个测试。

```bash
# 运行所有测试（耗时约 20 秒）
python -m unittest discover tests -v

# 运行单个测试文件
python -m unittest tests.test_models -v
python -m unittest tests.test_pretrain -v
python -m unittest tests.test_pretrain_finetune -v

# 运行单个测试方法
python -m unittest tests.test_models.TestModelOutputShape.test_all_models_output_7 -v
```

## 异常值处理与图表 Y 轴准确性

数据预处理阶段（`AirQualityDataProcessor._handle_outliers`）会执行 IQR 异常值裁剪：
- **AQI**：硬编码上限 500
- **CO**：硬编码上限 50
- **其他污染物**：`Q3 + 3×IQR` 统计上限

这一步是为了提升训练对极端值的鲁棒性，但会让回测图表（`plot_backtest_results`）的"真实值"曲线被裁剪后的范围所限制。`scripts/evaluate.py` 通过传入 `y_true_raw`（原始未裁剪的真值）解决：

```python
plot_backtest_results(
    y_true=result['y_true'],     # 经裁剪/缩放的真值（用于指标计算）
    y_pred=result['y_pred'],
    dates=result['dates'],
    y_true_raw=y_true_raw,       # 原始未裁剪真值（用于绘图，确保 y 轴准确）
)
```

未传 `y_true_raw` 时图表回退到 `y_true`（旧行为，向后兼容）。

## 性能基准

在北京 2015-2024 空气质量数据集上的测试结果（参考值，实际结果会随超参与随机种子变化）：

| 模型 | 验证损失 (Huber) | 训练时间 |
|------|------------------|----------|
| CNN | 0.1135 | ~48s |
| LSTM | 0.1161 | ~31000s |
| Hybrid | 0.1178 | ~540s |
| Transformer | 0.1246 | ~260s |

## 文档

- [算法架构分析](docs/algorithm_analysis.md) - 详细的算法原理和架构设计
- [CLAUDE.md](CLAUDE.md) - Claude Code 开发指南与架构决策

## 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 许可证

本项目仅供学习和研究使用。

## 致谢

- 数据来源：北京市空气质量监测数据
- 模型架构参考：Attention Is All You Need, LSTM, ResNet 等经典论文
