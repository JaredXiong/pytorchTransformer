"""
推理模块
"""
from .predictor import (
    AirQualityPredictor,
    PredictionResult,
    predict_air_quality,
    format_prediction_result,
)

__all__ = [
    'AirQualityPredictor',
    'PredictionResult',
    'predict_air_quality',
    'format_prediction_result',
]
