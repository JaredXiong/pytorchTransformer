"""
air_quality - 空气质量时间序列预测包

基于 14 天输入序列，预测未来 3 天的 AQI 及污染物浓度（PM2.5、PM10、NO2、SO2、CO、O3）。
"""
from .config import config
from .data import (
    AirQualityDataProcessor,
    DataStats,
    get_device,
    split_data,
    validate_prediction,
    calculate_metrics,
)
from .models import (
    BaseModel,
    TransformerModel,
    LSTMModel,
    CNNModel,
    HybridModel,
    EnsembleModel,
    create_model,
)
from .training import ModelTrainer, MetricsTracker, get_loss_function, WeightedMSELoss
from .inference import (
    AirQualityPredictor,
    PredictionResult,
    predict_air_quality,
    format_prediction_result,
)
from .visualization import (
    plot_training_history,
    plot_prediction_comparison,
    plot_feature_comparison,
    plot_metrics,
)

__version__ = "0.0.2"
__all__ = [
    # config
    "config",
    # data
    "AirQualityDataProcessor",
    "DataStats",
    "get_device",
    "split_data",
    "validate_prediction",
    "calculate_metrics",
    # models
    "BaseModel",
    "TransformerModel",
    "LSTMModel",
    "CNNModel",
    "HybridModel",
    "EnsembleModel",
    "create_model",
    # training
    "ModelTrainer",
    "MetricsTracker",
    "get_loss_function",
    "WeightedMSELoss",
    # inference
    "AirQualityPredictor",
    "PredictionResult",
    "predict_air_quality",
    "format_prediction_result",
    # visualization
    "plot_training_history",
    "plot_prediction_comparison",
    "plot_feature_comparison",
    "plot_metrics",
]
