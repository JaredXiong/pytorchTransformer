"""
配置模块 - 集中管理所有配置参数
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    """模型相关配置"""
    input_size: int = 9  # 输入特征数量
    d_model: int = 128  # Transformer隐藏层维度
    lstm_hidden: int = 256  # LSTM隐藏层维度
    cnn_filters: int = 128  # CNN过滤器数量
    nhead: int = 4  # 注意力头数量
    num_layers: int = 4  # 模型层数
    output_size: int = 7  # 预测目标维度（仅污染物，不含 month/season）
    dropout: float = 0.2  # Dropout概率


@dataclass
class TrainingConfig:
    """训练相关配置"""
    epochs: int = 250
    batch_size: int = 32
    learning_rate: float = 0.0003
    weight_decay: float = 3e-4
    gradient_clip: float = 1.0
    early_stop_patience: int = 60
    loss_type: str = 'huber'
    delta: float = 1.0


@dataclass
class DataConfig:
    """数据相关配置"""
    data_file: str = 'data/raw/北京2015-2024.xlsx'
    seq_length: int = 14  # 输入序列长度
    prediction_days: int = 3  # 预测天数
    train_split_ratio: float = 0.8  # 训练集比例
    feature_columns: List[str] = field(default_factory=lambda: [
        'aqi', 'pm2_5_24h', 'pm10_24h', 'no2_24h', 'so2_24h', 'co_24h', 'o3_8h_24h',
        'month', 'season'
    ])
    target_columns: List[str] = field(default_factory=lambda: [
        'aqi', 'pm2_5_24h', 'pm10_24h', 'no2_24h', 'so2_24h', 'co_24h', 'o3_8h_24h'
    ])
    target_feature: str = 'aqi'  # 预测目标
    # 滚动统计特征窗口大小（暂时禁用以避免维度不匹配）
    rolling_windows: List[int] = field(default_factory=lambda: [])


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


@dataclass
class HybridModelConfig:
    """混合模型相关配置"""
    d_model: int = 128  # 隐藏层维度
    nhead: int = 4  # 注意力头数量
    dropout: float = 0.2  # Dropout概率
    residual_weight: float = 0.3  # 残差权重


class Config:
    """统一配置类"""
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.file = FileConfig()
        self.prediction = PredictionConfig()
        self.hybrid = HybridModelConfig()

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
