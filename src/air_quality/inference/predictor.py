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
from ..data.vmd import VMDDecomposer
from ..data.vmd_features import apply_vmd_to_aqi, apply_vmd_to_features, POLLUTANT_ORDER
from ..models import create_model

logger = logging.getLogger(__name__)


# 7 个污染物对应的硬上限（与 validate_prediction 保持一致）
# AQI (idx 0) <= 500, CO (idx 5) <= 50；其他无限
HARD_MAX_PER_FEATURE = np.array(
    [500.0, np.inf, np.inf, np.inf, np.inf, 50.0, np.inf],
    dtype=np.float64,
)


def _soft_saturate(arr: np.ndarray, max_per_feature: np.ndarray, scale: float) -> np.ndarray:
    """对超过 max_per_feature 的部分做软饱和（不直接硬切）。

    当 excess = max(arr - max, 0) > 0 时，输出 = max - scale * log1p(excess / scale)
    性质：
      - excess = 0（输入在 max 以下）：输出 == 原值（恒等映射，不破坏正常范围）
      - excess > 0：输出 < max，且 excess 越大输出越低（趋近下界）
      - scale 越大，过渡越平；scale 越小，过渡越陡
    当 max 为 inf（无上限）时，原值直接返回。
    """
    arr = np.asarray(arr, dtype=np.float64)
    max_per_feature = np.asarray(max_per_feature, dtype=np.float64)
    if arr.shape[-1] != max_per_feature.shape[0]:
        # 维度不匹配：退化为硬裁剪，保证数值稳定
        clipped = arr.copy()
        for i, m in enumerate(max_per_feature):
            if np.isfinite(m):
                clipped[..., i] = np.minimum(clipped[..., i], m)
        return clipped
    out = arr.copy()
    for i, m in enumerate(max_per_feature):
        if not np.isfinite(m):
            continue
        excess = np.maximum(arr[..., i] - m, 0.0)
        # 仅对 excess > 0 处软饱和；excess == 0 处保留原值（恒等映射）
        soft = m - scale * np.log1p(excess / scale)
        out[..., i] = np.where(excess > 0, soft, arr[..., i])
    return out


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
        (-1.0, 1.0),     # month_sin (周期性编码)
        (-1.0, 1.0),     # month_cos (周期性编码)
        (-1.0, 1.0),     # season_sin (周期性编码)
        (-1.0, 1.0),     # season_cos (周期性编码)
    ]
    INPUT_COLUMNS = [
        'aqi', 'pm2_5_24h', 'pm10_24h', 'no2_24h', 'so2_24h',
        'co_24h', 'o3_8h_24h', 'month_sin', 'month_cos', 'season_sin', 'season_cos',
    ]

    def __init__(self, model_weights_path: str = None, scaler_path: str = None):
        self.model_weights_path = model_weights_path or config.file.model_save_path
        self.scaler_path = scaler_path or config.file.scaler_path
        self.device = get_device()
        self.model = None
        self.scaler = None
        # VMD-CNN-BiLSTM-Attention 模型需要 VMDDecomposer 重建 12 维输入
        self._vmd_decomposer: VMDDecomposer | None = None
        # VMD target: 'aqi' 仅分解 AQI；'all' 分解全部 7 个污染物
        self._vmd_target: str = 'aqi'

    def load_model(self) -> None:
        if not os.path.exists(self.model_weights_path):
            raise FileNotFoundError(f"模型权重文件不存在: {self.model_weights_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"标准化器文件不存在: {self.scaler_path}")

        self.scaler = joblib.load(self.scaler_path)

        try:
            checkpoint = torch.load(self.model_weights_path, map_location=self.device)
            model_type = self._infer_model_type(checkpoint)

            # VMD 半监督训练会把 9 维特征扩展为 8+K 维，所以 input_size
            # 必须从 checkpoint 读取，config 中的默认 9 维不能用作兜底。
            input_size = checkpoint['input_size']
            output_size = checkpoint['output_size']

            # VMD 模型需传入 vmd_K，避免 pretrain_head 维度不匹配
            create_kwargs = {}
            if model_type == 'vmd_cnn_bilstm_attention' and 'vmd_params' in checkpoint:
                create_kwargs['vmd_K'] = checkpoint['vmd_params']['K']

            self.model = create_model(
                model_type=model_type,
                input_size=input_size,
                output_size=output_size,
                **create_kwargs,
            ).to(self.device)

            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.model.load_state_dict(state_dict)
            self.model.eval()

            # VMD 模型：重建 VMDDecomposer 实例 + 记录 vmd_target
            if model_type == 'vmd_cnn_bilstm_attention':
                self._vmd_decomposer = VMDDecomposer(**checkpoint['vmd_params'])
                # 读取训练时的 vmd_target；缺省时根据 input_size 自动推断
                self._vmd_target = checkpoint.get('vmd_target', 'aqi')
                # 兜底：如果 checkpoint 没存 vmd_target，用 input_size 反推
                if 'vmd_target' not in checkpoint:
                    K = self._vmd_decomposer.K
                    expected_aqi = K + 8  # K IMFs + 6 pollutants + 2 calendar
                    expected_all = 7 * K + 2  # 7*K IMFs + 2 calendar
                    if input_size == expected_all:
                        self._vmd_target = 'all'
                    elif input_size == expected_aqi:
                        self._vmd_target = 'aqi'
        except RuntimeError as e:
            raise ValueError(f"模型权重加载失败: {str(e)}")

    @staticmethod
    def _infer_model_type(checkpoint: dict) -> str:
        raw = str(checkpoint.get('model_type', 'transformer'))
        lowered = raw.lower()
        # 复合模型必须在简单模型之前匹配，避免 "bilstm" 中的 "lstm" 误匹配
        if 'vmd' in lowered or 'bilstm' in lowered:
            return 'vmd_cnn_bilstm_attention'
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
        """将原始 (14, 9) 输入转换为模型所需的 (1, 14, F) 张量。

        非 VMD 模型：F=9。缩放只作用于前 7 列污染物；后 2 列 (month, season) 保持整数原值。
        VMD-CNN-BiLSTM-Attention 模型：F=8+K=12。前 7 列污染物做 scaler 缩放后，
        AQI(idx 0) 被 4 个 IMF 替换，pm2_5..o3(idx 1..6) 保留，最后拼 month/season。
        """
        # validate_input 假设原始尺度 (0-500 等)。forecast 用它做用户输入校验。
        self.validate_input(input_sequence)
        return self._build_model_tensor(input_sequence)

    def _vmd_transform(self, poll_scaled: np.ndarray) -> np.ndarray:
        """对 (T, 7) 缩放后的污染物做 VMD 特征变换，返回 (T, F_vmd)。

        根据 ``self._vmd_target`` 选择两种变换路径：
        - 'aqi'  (默认)：仅分解 AQI 列，返回 K IMFs + 6 个其他污染物 (T, K+6)
        - 'all' (推荐)：分解全部 7 个污染物，返回 7*K IMFs (T, 7*K)
        """
        if self._vmd_target == 'all':
            K = self._vmd_decomposer.K
            # poll_scaled.T → (7, T)；decompose 返回 (7, K, T)
            imfs = self._vmd_decomposer.decompose(poll_scaled.T)
            # 重排为 (T, 7, K) 再展平为 (T, 7*K)
            imfs = imfs.transpose(0, 2, 1).reshape(poll_scaled.shape[0], -1)
            return imfs
        # 默认 'aqi' 模式
        K = self._vmd_decomposer.K
        aqi_scaled = poll_scaled[:, 0]  # (T,)
        imfs = self._vmd_decomposer.decompose(aqi_scaled)  # (K, T)
        imfs = imfs.T  # (T, K)
        other_pollutants = poll_scaled[:, 1:7]  # (T, 6)
        return np.hstack([imfs, other_pollutants])  # (T, K+6)

    def _build_model_tensor(self, input_sequence: np.ndarray) -> torch.Tensor:
        """纯数据转换：(T, 9) → (1, T, F)。**不做** validate_input。

        与 _prepare_model_input 的区别：本函数**不验证**输入范围，
        供 backtest 等已经在缩放后数据上工作的调用方使用，避免被误判为越界。
        """
        scaled = self.scaler.transform(input_sequence[:, :7])  # (T, 7)
        if self._vmd_decomposer is None:
            model_input = np.hstack([scaled, input_sequence[:, 7:]])
        else:
            vmd_features = self._vmd_transform(scaled)  # (T, K+6) 或 (T, 7K)
            model_input = np.hstack([vmd_features, input_sequence[:, 7:]])
        return torch.FloatTensor(model_input).unsqueeze(0).to(self.device)

    def forecast(self, input_sequence: np.ndarray,
                 future_dates: list = None) -> dict:
        """生产模式：基于最近 14 天数据预测未来 N 天。"""
        if self.model is None or self.scaler is None:
            raise ValueError("模型未加载，请先调用 load_model()")

        input_tensor = self._prepare_model_input(input_sequence)
        num_days = config.data.prediction_days

        if self._vmd_target == 'all':
            # VMD(target='all') 不支持 K>T=1 的单步重投影，改为一次性 forward 取末尾 N 步
            with torch.no_grad():
                output = self.model(input_tensor)  # (1, T, 7)
                predictions = output[:, -num_days:, :].cpu().numpy()[0]
        else:
            # 自回归模式：每步必须用正确 calendar，否则模型被迫按 calendar 预测，
            # 会出现"阶梯线"现象（calendar 跨年突变时预测值突变）。
            # 优先用 future_dates 推算周期性编码，否则以窗口最后一帧为起点递增。
            if future_dates is not None and len(future_dates) >= num_days:
                fc_list = []
                for fd in future_dates[:num_days]:
                    dt = pd.to_datetime(fd)
                    month = dt.month
                    season = (month % 12 + 3) // 3
                    # 周期性编码：month 和 season 各用 sin/cos 表示
                    month_sin = np.sin(2 * np.pi * month / 12)
                    month_cos = np.cos(2 * np.pi * month / 12)
                    season_sin = np.sin(2 * np.pi * season / 4)
                    season_cos = np.cos(2 * np.pi * season / 4)
                    fc_list.append([month_sin, month_cos, season_sin, season_cos])
                future_calendar = torch.FloatTensor(fc_list).unsqueeze(0).to(self.device)
            else:
                # 从输入序列的最后 4 个特征（周期性编码）推断未来的 calendar
                last_cal = input_tensor[:, -1, -4:].clone()  # (1, 4) = [month_sin, month_cos, season_sin, season_cos]
                # 通过 arctan2 恢复原始月份，然后递增
                last_month_sin, last_month_cos = last_cal[0, 0].item(), last_cal[0, 1].item()
                last_month = int(np.arctan2(last_month_sin, last_month_cos) * 12 / (2 * np.pi)) % 12 + 1
                # 生成未来 calendar
                fc_list = []
                for step in range(num_days):
                    future_month = ((last_month - 1 + step + 1) % 12) + 1
                    future_season = (future_month % 12 + 3) // 3
                    month_sin = np.sin(2 * np.pi * future_month / 12)
                    month_cos = np.cos(2 * np.pi * future_month / 12)
                    season_sin = np.sin(2 * np.pi * future_season / 4)
                    season_cos = np.cos(2 * np.pi * future_season / 4)
                    fc_list.append([month_sin, month_cos, season_sin, season_cos])
                future_calendar = torch.FloatTensor(fc_list).unsqueeze(0).to(self.device)

            predictions = []
            with torch.no_grad():
                current_input = input_tensor.clone()
                for step in range(num_days):
                    output = self.model(current_input)
                    pred = output[:, -1, :]
                    predictions.append(pred.cpu().numpy()[0])
                    last_step = current_input[:, -1:, :].clone()
                    if self._vmd_decomposer is None:
                        last_step[:, :, :7] = pred.unsqueeze(1)
                        last_step[:, :, 7:11] = future_calendar[:, step:step + 1, :]  # 4 个周期性编码特征
                    else:  # 'aqi'
                        K = self._vmd_decomposer.K
                        last_step[:, :, K:K + 6] = pred.unsqueeze(1)[:, :, 1:7]
                        last_step[:, :, -4:] = future_calendar[:, step:step + 1, :]  # 4 个周期性编码特征
                    current_input = torch.cat([current_input[:, 1:, :], last_step], dim=1)
            predictions = np.array(predictions)
        predictions = self.scaler.inverse_transform(predictions)
        for i in range(predictions.shape[0]):
            predictions[i] = validate_prediction(predictions[i])
        # 反平滑：软化硬裁剪（默认开启）。硬下限 0 仍保留（物理合理性），
        # 硬上限（AQI=500、CO=50）改为软饱和，由 validate_prediction 在下一步硬保证。
        if config.prediction.soft_clip:
            predictions = _soft_saturate(
                predictions,
                max_per_feature=HARD_MAX_PER_FEATURE,
                scale=config.prediction.soft_clip_scale,
            )
        predictions = np.maximum(predictions, 0)  # 物理下界 0 仍硬保证

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
                # 走纯数据转换：backtest 输入是缩放后 9 维，不做原始尺度 validate。
                if self._vmd_decomposer is not None:
                    window_tensor = self._build_model_tensor(window)
                else:
                    window_tensor = torch.FloatTensor(window).unsqueeze(0).to(self.device)

                # 准备预测窗口对应的真实 calendar（周期性编码，4 个特征）：
                # 旧实现 3 天预测共用窗口最后一帧的 calendar，导致模型被迫按
                # calendar 预测、跨年时预测值突变（"阶梯线"现象）。
                # 正确做法：每步用对应预测日的真实 calendar。
                future_calendar = data[start + seq_len:start + seq_len + prediction_days, 7:11]
                future_calendar = torch.FloatTensor(future_calendar).unsqueeze(0).to(self.device)

                preds = []
                cur = window_tensor.clone()
                for step in range(prediction_days):
                    out = self.model(cur)
                    p = out[:, -1, :]
                    preds.append(p.cpu().numpy()[0])
                    # 自回归：根据 _vmd_target 决定如何替换 last_step
                    last_step = cur[:, -1:, :].clone()
                    if self._vmd_decomposer is None:
                        last_step[:, :, :7] = p.unsqueeze(1)
                        # 关键修复：用对应预测日的真实 calendar 替换，避免 calendar 信息泄露
                        last_step[:, :, 7:11] = future_calendar[:, step:step + 1, :]  # 4 个周期性编码特征
                    else:  # 'aqi'
                        K = self._vmd_decomposer.K
                        last_step[:, :, K:K + 6] = p.unsqueeze(1)[:, :, 1:7]
                        last_step[:, :, -4:] = future_calendar[:, step:step + 1, :]  # 4 个周期性编码特征
                    cur = torch.cat([cur[:, 1:, :], last_step], dim=1)

                preds = np.array(preds)
                preds = self.scaler.inverse_transform(preds)
                for i in range(preds.shape[0]):
                    preds[i] = validate_prediction(preds[i])
                # 反平滑：与 forecast 保持一致，应用软饱和（默认开启）。
                # 物理下界 0 仍由下一行硬保证。
                if config.prediction.soft_clip:
                    preds = _soft_saturate(
                        preds,
                        max_per_feature=HARD_MAX_PER_FEATURE,
                        scale=config.prediction.soft_clip_scale,
                    )
                preds = np.maximum(preds, 0)

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
                        num_days: int = None,
                        future_dates: list = None) -> dict:
    """便捷预测函数 — 等价于 AirQualityPredictor.forecast()。"""
    if input_sequence is None:
        raise ValueError("必须提供输入序列数据")

    predictor = AirQualityPredictor(model_weights_path, scaler_path)
    predictor.load_model()
    return predictor.forecast(input_sequence, future_dates=future_dates)


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
