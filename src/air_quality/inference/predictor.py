"""
推理模块 - 加载训练好的模型并执行自回归预测
"""
import os
import logging
from typing import Dict, Any
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import joblib

from ..config import config
from ..data import get_device, validate_prediction, calculate_metrics
from ..models import create_model

logger = logging.getLogger(__name__)


class PredictionResult:
    """预测结果封装类"""

    AQI_GRADES = {
        (0, 50): "优",
        (51, 100): "良",
        (101, 150): "轻度污染",
        (151, 200): "中度污染",
        (201, 300): "重度污染",
        (301, float('inf')): "严重污染",
    }

    FEATURE_NAMES = {
        'aqi': 'AQI',
        'pm2_5_24h': 'PM2.5',
        'pm10_24h': 'PM10',
        'no2_24h': 'NO2',
        'so2_24h': 'SO2',
        'co_24h': 'CO',
        'o3_8h_24h': 'O3',
    }

    @classmethod
    def get_grade(cls, aqi: float) -> str:
        """根据AQI值获取空气质量等级"""
        for (min_val, max_val), grade in cls.AQI_GRADES.items():
            if min_val <= aqi <= max_val:
                return grade
        return "未知"


class AirQualityPredictor:
    """空气质量预测器 - 支持 backtest（评估）与 forecast（生产）两种模式。"""

    INPUT_RANGES = [
        (0.0, 500.0),    # aqi
        (0.0, 1000.0),   # pm2_5_24h
        (0.0, 1000.0),   # pm10_24h
        (0.0, 500.0),    # no2_24h
        (0.0, 500.0),    # so2_24h
        (0.0, 50.0),     # co_24h
        (0.0, 500.0),    # o3_8h_24h
        (1, 12),         # month (int)
        (1, 4),          # season (int)
    ]
    INPUT_COLUMNS = [
        'aqi', 'pm2_5_24h', 'pm10_24h', 'no2_24h', 'so2_24h',
        'co_24h', 'o3_8h_24h', 'month', 'season',
    ]

    def __init__(self, model_weights_path: str = None, scaler_path: str = None):
        self.model_weights_path = model_weights_path or config.file.model_save_path
        self.scaler_path = scaler_path or config.file.scaler_path
        self.device = get_device()
        self.model = None
        self.scaler = None

    def load_model(self) -> None:
        if not os.path.exists(self.model_weights_path):
            raise FileNotFoundError(f"模型权重文件不存在: {self.model_weights_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"标准化器文件不存在: {self.scaler_path}")

        self.scaler = joblib.load(self.scaler_path)

        try:
            checkpoint = torch.load(self.model_weights_path, map_location=self.device)
            model_type = self._infer_model_type(checkpoint)

            self.model = create_model(
                model_type=model_type,
                input_size=config.model.input_size,
                output_size=config.model.output_size,
            ).to(self.device)

            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        except RuntimeError as e:
            raise ValueError(f"模型权重加载失败: {str(e)}")

    @staticmethod
    def _infer_model_type(checkpoint: dict) -> str:
        raw = str(checkpoint.get('model_type', 'transformer'))
        lowered = raw.lower()
        if 'hybrid' in lowered:
            return 'hybrid'
        if 'transformer' in lowered:
            return 'transformer'
        if 'gru' in lowered:
            return 'gru'
        if 'lstm' in lowered:
            return 'lstm'
        if 'tcn' in lowered:
            return 'tcn'
        if 'cnn' in lowered:
            return 'cnn'
        return 'transformer'

    @classmethod
    def validate_input(cls, input_sequence: np.ndarray) -> None:
        """验证输入形状和每列取值范围。"""
        if not isinstance(input_sequence, np.ndarray):
            raise TypeError("输入序列必须是numpy数组")
        if input_sequence.ndim != 2:
            raise ValueError(f"输入序列维度错误，应为2维，实际为{input_sequence.ndim}维")
        if input_sequence.shape != (config.data.seq_length, config.model.input_size):
            raise ValueError(
                f"输入形状错误，应为 ({config.data.seq_length}, {config.model.input_size})，"
                f"实际为 {input_sequence.shape}"
            )
        for i, (lo, hi) in enumerate(cls.INPUT_RANGES):
            col = input_sequence[:, i]
            if col.min() < lo or col.max() > hi:
                raise ValueError(
                    f"输入列 {cls.INPUT_COLUMNS[i]} (idx {i}) 越界: "
                    f"应为 [{lo}, {hi}]，实际 min={col.min():.4f}, max={col.max():.4f}"
                )

    def _prepare_model_input(self, input_sequence: np.ndarray) -> torch.Tensor:
        """将原始 (14, 9) 输入转换为模型所需的 (1, 14, 9) 张量。

        缩放只作用于前 7 列污染物；后 2 列 (month, season) 保持整数原值。
        """
        self.validate_input(input_sequence)
        scaled = self.scaler.transform(input_sequence[:, :7])
        model_input = np.hstack([scaled, input_sequence[:, 7:]])
        return torch.FloatTensor(model_input).unsqueeze(0).to(self.device)

    def forecast(self, input_sequence: np.ndarray,
                 future_dates: list = None) -> dict:
        """生产模式：基于最近 14 天数据预测未来 N 天。"""
        if self.model is None or self.scaler is None:
            raise ValueError("模型未加载，请先调用 load_model()")

        input_tensor = self._prepare_model_input(input_sequence)
        num_days = config.data.prediction_days

        predictions = []
        with torch.no_grad():
            current_input = input_tensor.clone()
            for _ in range(num_days):
                output = self.model(current_input)
                pred = output[:, -1, :]
                predictions.append(pred.cpu().numpy()[0])
                # Splice 7-dim prediction back into 9-dim window (keep month/season from last step)
                last_step = current_input[:, -1:, :].clone()
                last_step[:, :, :7] = pred.unsqueeze(1)
                current_input = torch.cat([current_input[:, 1:, :], last_step], dim=1)

        predictions = np.array(predictions)
        predictions = self.scaler.inverse_transform(predictions)
        for i in range(predictions.shape[0]):
            predictions[i] = validate_prediction(predictions[i])
        predictions = np.clip(predictions, a_min=0, a_max=500)

        if future_dates is None:
            future_dates = [(datetime.now() + timedelta(days=i + 1)).strftime('%Y-%m-%d')
                            for i in range(num_days)]

        return {
            'predictions': predictions,
            'dates': future_dates,
            'aqi_grades': [PredictionResult.get_grade(pred[0]) for pred in predictions],
            'features': list(self.INPUT_COLUMNS[:7]),
        }

    def backtest(self, data: np.ndarray, dates,
                 prediction_days: int = 3) -> dict:
        """评估模式：对全数据集做滚动窗口预测，与真实值对比。"""
        if self.model is None or self.scaler is None:
            raise ValueError("模型未加载，请先调用 load_model()")

        seq_len = config.data.seq_length
        n = len(data)
        n_samples = n - seq_len - prediction_days + 1
        if n_samples <= 0:
            raise ValueError(f"数据长度 {n} 不足以做回测 (需要 ≥ {seq_len + prediction_days + 1})")

        y_true_list = []
        y_pred_list = []
        date_list = []

        dates_dt = pd.to_datetime(dates).reset_index(drop=True)

        with torch.no_grad():
            for start in range(n_samples):
                window = data[start:start + seq_len]
                window_tensor = torch.FloatTensor(window).unsqueeze(0).to(self.device)

                preds = []
                cur = window_tensor.clone()
                for _ in range(prediction_days):
                    out = self.model(cur)
                    p = out[:, -1, :]
                    preds.append(p.cpu().numpy()[0])
                    last_step = cur[:, -1:, :].clone()
                    last_step[:, :, :7] = p.unsqueeze(1)
                    cur = torch.cat([cur[:, 1:, :], last_step], dim=1)

                preds = np.array(preds)
                preds = self.scaler.inverse_transform(preds)
                for i in range(preds.shape[0]):
                    preds[i] = validate_prediction(preds[i])

                target_idx = slice(start + seq_len, start + seq_len + prediction_days)
                true_scaled = data[target_idx, :7]
                true_unscaled = self.scaler.inverse_transform(true_scaled)

                y_pred_list.append(preds)
                y_true_list.append(true_unscaled)
                date_list.append([
                    dates_dt.iloc[start + seq_len + i].strftime('%Y-%m-%d')
                    for i in range(prediction_days)
                ])

        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)

        metrics = calculate_metrics(y_true, y_pred)

        return {
            'y_true': y_true,
            'y_pred': y_pred,
            'dates': date_list,
            'metrics': metrics,
        }


def predict_air_quality(model_weights_path: str = None,
                        scaler_path: str = None,
                        input_sequence: np.ndarray = None,
                        num_days: int = None) -> dict:
    """便捷预测函数 — 等价于 AirQualityPredictor.forecast()。"""
    if input_sequence is None:
        raise ValueError("必须提供输入序列数据")

    predictor = AirQualityPredictor(model_weights_path, scaler_path)
    predictor.load_model()
    return predictor.forecast(input_sequence)


def format_prediction_result(result: Dict[str, Any], show_details: bool = True) -> str:
    """格式化预测结果为可读字符串"""
    output_lines = []
    output_lines.append("空气质量预测结果")
    output_lines.append("=" * 50)

    dates = result.get('dates', [])
    predictions = result.get('predictions', [])
    aqi_grades = result.get('aqi_grades', [])
    features = result.get('features', [])

    # 每天的AQI和等级
    for i, date in enumerate(dates):
        if aqi_grades and i < len(aqi_grades):
            grade = aqi_grades[i]
            aqi_value = "N/A"
            if 'aqi' in features and i < len(predictions) and features.index('aqi') < len(predictions[i]):
                aqi_idx = features.index('aqi')
                aqi_value = f"{predictions[i][aqi_idx]:.1f}"
            output_lines.append(f"{date}: AQI = {aqi_value}, 等级 = {grade}")
        else:
            output_lines.append(f"{date}: 预测完成")

    # 详细污染指标
    if show_details and len(predictions) > 0 and len(features) > 0:
        output_lines.append("\n详细污染指标:")
        output_lines.append("-" * 50)

        header = "日期"
        for feature in features:
            feature_name = PredictionResult.FEATURE_NAMES.get(feature, feature)
            header += f" | {feature_name:^10}"
        output_lines.append(header)
        output_lines.append("-" * 50)

        for i, (date, pred) in enumerate(zip(dates, predictions)):
            row = date
            for j, value in enumerate(pred[:len(features)]):
                if features[j] in ['co_24h']:
                    row += f" | {value:^10.2f}"
                else:
                    row += f" | {value:^10.1f}"
            output_lines.append(row)

    return "\n".join(output_lines)
