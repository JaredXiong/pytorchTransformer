"""
配置模块 - 集中管理所有配置参数
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    """模型相关配置"""
    input_size: int = 11  # 输入特征数量（7 污染物 + 4 周期性日历特征）
    d_model: int = 128  # Transformer隐藏层维度
    lstm_hidden: int = 256  # LSTM隐藏层维度
    cnn_filters: int = 128  # CNN过滤器数量
    nhead: int = 4  # 注意力头数量
    num_layers: int = 4  # 模型层数
    output_size: int = 7  # 预测目标维度（仅污染物，不含 month/season）
    dropout: float = 0.1  # Dropout概率（已降低以减弱均值回归倾向）
    # Transformer 反平滑专用配置
    transformer_residual_weight: float = 0.3  # 残差 shortcut 初始权重
    transformer_feature_scale_init: float = 1.0  # feature_scale 初始值
    transformer_ffn_mult: int = 4  # dim_feedforward = ffn_mult * d_model
    transformer_norm_first: bool = True  # 显式 pre-LN，稳定深层训练


@dataclass
class TrainingConfig:
    """训练相关配置"""
    epochs: int = 250
    batch_size: int = 32
    learning_rate: float = 0.0003
    weight_decay: float = 1e-4  # 已降低（从 3e-4）以减弱对低幅权重的偏好
    gradient_clip: float = 1.0
    early_stop_patience: int = 40  # 已降低（从 60）以在保守解之前停下
    loss_type: str = 'mse_antismooth'  # 已从 huber 切换到反平滑复合损失
    delta: float = 1.0
    # 反平滑损失分量（温和增强，保留修复效果但不过激）
    lambda_var: float = 0.15   # 温和上调（原 0.1），压制 MSE 主导地位但不过激
    lambda_diff: float = 0.08  # 温和上调（原 0.05）
    tau_var: float = 0.55      # 温和上调（原 0.5），要求 pred_std ≥ 55% target_std
    tau_diff: float = 0.55     # 温和上调（原 0.5）
    lambda_warmup_epochs: int = 20
    # 反平滑早停信号（threshold 已从 0.1 上调到 0.3，更早触发）
    detect_smoothing: bool = True
    smoothing_threshold: float = 0.3
    smoothing_stop_patience: int = 15


@dataclass
class DataConfig:
    """数据相关配置"""
    data_file: str = 'data/raw/北京2015-2024.xlsx'
    seq_length: int = 14  # 输入序列长度
    prediction_days: int = 3  # 预测天数
    train_split_ratio: float = 0.8  # 训练集比例
    feature_columns: List[str] = field(default_factory=lambda: [
        'aqi', 'pm2_5_24h', 'pm10_24h', 'no2_24h', 'so2_24h', 'co_24h', 'o3_8h_24h',
        'month_sin', 'month_cos', 'season_sin', 'season_cos'
    ])
    target_columns: List[str] = field(default_factory=lambda: [
        'aqi', 'pm2_5_24h', 'pm10_24h', 'no2_24h', 'so2_24h', 'co_24h', 'o3_8h_24h'
    ])
    target_feature: str = 'aqi'  # 预测目标
    # 滚动统计特征窗口大小（暂时禁用以避免维度不匹配）
    rolling_windows: List[int] = field(default_factory=lambda: [])
    # scaler 拟合阶段：True=在未裁剪数据上 fit（保留真实动态范围，clip 仅作用于训练目标）
    scaler_on_uncapped: bool = True


@dataclass
class FileConfig:
    """文件路径配置"""
    model_save_path: str = 'outputs/checkpoints/air_quality_model.pth'
    scaler_path: str = 'outputs/checkpoints/scaler.pkl'
    output_dir: str = 'outputs'
    figures_dir: str = 'outputs/figures'


@dataclass
class PredictionConfig:
    """预测相关配置"""
    min_prediction_value: float = 0.0
    max_prediction_value: float = 500.0
    feature_ratio_constraint: float = 1.0
    # 软饱和：避免硬裁剪 0-500 二次压缩预测动态范围
    soft_clip: bool = True
    soft_clip_scale: float = 20.0


@dataclass
class HybridModelConfig:
    """混合模型相关配置"""
    d_model: int = 128  # 隐藏层维度
    nhead: int = 4  # 注意力头数量
    dropout: float = 0.1  # Dropout概率（与 ModelConfig.dropout 对齐）
    residual_weight: float = 0.3  # 残差权重


@dataclass
class VMDConfig:
    """VMD 分解配置"""
    enabled: bool = True
    K: int = 3                       # IMF 模态数（从 4 降到 3，因为 target='all' 时
                                     # 输入维度 = 7*K + 2；K=4 时 30 维过大）
    alpha: float = 2000              # 带宽约束
    tau: float = 0                   # 噪声容忍
    DC: int = 0                      # 是否保留 DC
    init: int = 1                    # 初始化方式
    tol: float = 1e-7                # 收敛容差
    # VMD 目标：'aqi' = 只对 AQI 分解（旧行为，向后兼容）；
    #           'all' = 对全部 7 个污染物分解（实验性，未验证带来收益）
    target: str = 'aqi'


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


class Config:
    """统一配置类"""
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.file = FileConfig()
        self.prediction = PredictionConfig()
        self.hybrid = HybridModelConfig()
        self.vmd = VMDConfig()
        self.semi = SemiSupervisedConfig()
        self.pretrain = PretrainConfig()

    def update_from_dict(self, config_dict: dict):
        """从字典更新配置"""
        for section, params in config_dict.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                for key, value in params.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)


# 创建全局配置实例
config = Config()
