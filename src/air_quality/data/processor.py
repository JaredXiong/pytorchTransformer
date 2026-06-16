"""
数据处理模块 - 数据加载、预处理、特征工程
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import torch
import logging
import os
from typing import Tuple, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DataStats:
    """数据统计信息"""
    original_shape: Tuple[int, int]
    processed_shape: Tuple[int, int]
    feature_names: List[str]
    train_samples: int
    test_samples: int
    date_range: Tuple[str, str]


class AirQualityDataProcessor:
    """空气质量数据处理器 - 支持滚动统计特征"""

    def __init__(self, feature_columns: List[str] = None, seq_length: int = 14,
                 prediction_days: int = 3, rolling_windows: List[int] = None,
                 output_size: int = 7, scaler_on_uncapped: bool = True):
        self.feature_columns = feature_columns or [
            'aqi', 'pm2_5_24h', 'pm10_24h', 'no2_24h', 'so2_24h', 'co_24h', 'o3_8h_24h',
            'month', 'season'
        ]
        self.output_size = output_size
        self.seq_length = seq_length
        self.prediction_days = prediction_days
        self.rolling_windows = rolling_windows if rolling_windows is not None else []
        self.scaler_on_uncapped = scaler_on_uncapped
        self.scaler = None
        self.imputer = None
        self.stats: Optional[DataStats] = None

    def load_and_preprocess_data(self, file_path: str) -> Tuple[np.ndarray, StandardScaler, List[str], pd.Series]:
        """加载并预处理空气质量数据。

        返回的 data 形状为 (N, 9)：前 7 列为缩放后的污染物，后 2 列为原始 month/season 整数。
        scaler 仅拟合 7 个污染物特征。

        反平滑（scaler_on_uncapped=True，默认）：scaler 在 **未裁剪** 数据上 fit，
        clip 仅作用于后续 create_sequences 的训练目标，保留真实分布的动态范围。
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        df = pd.read_excel(file_path, engine='openpyxl')
        original_shape = df.shape

        # 日期处理
        date_col = 'pubtime' if 'pubtime' in df.columns else 'date'
        dates = df[date_col].copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        # 提取时间特征
        df['month'] = df[date_col].dt.month
        df['season'] = (df[date_col].dt.month % 12 + 3) // 3

        # 列名映射
        column_mapping = self._create_column_mapping(df.columns)
        df = df.rename(columns=column_mapping)

        # 筛选可用污染物特征（不含 month/season）
        pollutant_features = [c for c in self.feature_columns
                              if c in df.columns and c not in ('month', 'season')]
        if len(pollutant_features) < 3:
            raise ValueError(f"可用污染物特征不足，仅找到 {len(pollutant_features)} 个特征")

        # 污染物数据预处理
        raw_poll_data = df[pollutant_features].values

        # 拆分裁剪：scaler_on_uncapped=True 时保留一份未裁剪数据用于 fit scaler
        if self.scaler_on_uncapped:
            # 对 NaN/缺失值先做一次中位数填补（仅用于 scaler fit），不影响后续裁剪+impute 路径
            from sklearn.impute import SimpleImputer as _SI
            scaler_fit_data = _SI(strategy='median').fit_transform(raw_poll_data)

        # clip + impute 主路径
        clipped_poll_data = self._handle_outliers(raw_poll_data, pollutant_features)
        self.imputer = SimpleImputer(strategy='mean')
        poll_data = self.imputer.fit_transform(clipped_poll_data)

        # 滚动统计特征（仅作用于污染物，默认关闭）
        poll_df = pd.DataFrame(poll_data, columns=pollutant_features)
        rolling_features = []
        for window in self.rolling_windows:
            for col in pollutant_features:
                rolling_mean = poll_df[col].rolling(window=window, min_periods=1).mean()
                rolling_std = poll_df[col].rolling(window=window, min_periods=1).std().fillna(0)
                rolling_features.append(rolling_mean.values)
                rolling_features.append(rolling_std.values)

        if rolling_features:
            rolling_data = np.column_stack(rolling_features)
            poll_data = np.hstack([poll_data, rolling_data])

        # 标准化（仅对污染物）
        # 关键：scaler 在未裁剪数据上 fit（如果启用），让 inverse_transform 保留宽动态范围
        self.scaler = StandardScaler()
        if self.scaler_on_uncapped:
            self.scaler.fit(scaler_fit_data)
        scaled_pollutants = self.scaler.transform(poll_data)

        # 拼接日历特征（month/season 原始整数，不缩放）
        calendar = df[['month', 'season']].values
        data = np.hstack([scaled_pollutants, calendar])

        # 记录 scaler 的特征名（用 pollutant_features，未含滚动列名以保持向后兼容）
        self.scaler.feature_names_in_ = list(poll_df.columns) + [
            f'rolling_{w}' for w in self.rolling_windows
            for _ in range(len(pollutant_features) * 2)
        ]

        self.stats = DataStats(
            original_shape=original_shape,
            processed_shape=data.shape,
            feature_names=pollutant_features,
            train_samples=0,
            test_samples=0,
            date_range=(str(dates.iloc[0]), str(dates.iloc[-1]))
        )

        return data, self.scaler, pollutant_features, dates

    def _create_column_mapping(self, columns) -> dict:
        """创建列名映射字典"""
        import re
        column_mapping = {}
        mapping_rules = [
            ('aqi', ['aqi', '空气质量指数']),
            ('pm2_5_24h', ['pm2.5', 'pm25', '细颗粒物']),
            ('pm10_24h', ['pm10', '可吸入颗粒物']),
            ('no2_24h', ['no2', '二氧化氮']),
            ('so2_24h', ['so2', '二氧化硫']),
            ('co_24h', ['co', '一氧化碳']),
            ('o3_8h_24h', ['o3', '臭氧']),
        ]
        for target_name, patterns in mapping_rules:
            for col in columns:
                col_lower = col.lower().replace(' ', '').replace('_', '')
                for pattern in patterns:
                    pattern_norm = pattern.lower().replace(' ', '')
                    if re.search(r'\b' + re.escape(pattern_norm), col_lower):
                        if col not in column_mapping:
                            column_mapping[col] = target_name
                        break
        return column_mapping

    def _handle_outliers(self, data: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """处理数据中的异常值

        Returns:
            handled_data: 与 data 同形状、按列 clip 后的数组
        """
        handled_data = data.copy()
        for i, feature in enumerate(feature_names):
            if not np.issubdtype(data[:, i].dtype, np.number):
                continue
            Q1 = np.nanpercentile(data[:, i], 25)
            Q3 = np.nanpercentile(data[:, i], 75)
            IQR = Q3 - Q1
            lower_bound = max(0, Q1 - 1.5 * IQR)
            if 'aqi' in feature.lower():
                upper_bound = 500
            elif 'co' in feature.lower():
                upper_bound = 50
            else:
                upper_bound = Q3 + 3.0 * IQR
            handled_data[:, i] = np.clip(data[:, i], lower_bound, upper_bound)
        return handled_data

    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """构造输入输出序列。

        X 形状 (N_samples, seq_length, 9)：包含 month/season 在最后两列。
        y 形状 (N_samples, prediction_days, 7)：仅污染物（前 7 列）。
        """
        min_required_length = self.seq_length + self.prediction_days
        if len(data) < min_required_length:
            raise ValueError(f"数据长度不足，需要至少 {min_required_length} 个样本")

        target_dim = self.output_size
        xs, ys = [], []
        total_length = len(data) - self.seq_length - self.prediction_days + 1
        for i in range(total_length):
            x = data[i:i + self.seq_length]
            y = data[i + self.seq_length:i + self.seq_length + self.prediction_days, :target_dim]
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)


def get_device() -> torch.device:
    """检测并返回合适的设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def validate_prediction(prediction: np.ndarray) -> np.ndarray:
    """验证预测结果的物理合理性（仅作用于 7 维污染物输出）。"""
    min_values = np.array([0, 0, 0, 0, 0, 0, 0])
    if prediction.shape[-1] >= 7:
        validated = np.maximum(prediction, min_values[:prediction.shape[-1]])
    else:
        validated = np.maximum(prediction, min_values[:prediction.shape[-1]])

    # AQI (idx 0) <= 500, CO (idx 5) <= 50
    validated[..., 0] = np.minimum(validated[..., 0], 500)
    if validated.shape[-1] >= 6:
        validated[..., 5] = np.minimum(validated[..., 5], 50)

    return validated


def split_data(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """划分训练集和测试集"""
    if len(X) != len(y):
        raise ValueError("输入数据和目标数据长度不匹配")
    split_idx = int(len(X) * train_ratio)
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, feature_names: list = None) -> dict:
    """计算多指标评估指标（默认 7 个污染物特征，不含 month/season）。"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    feature_names = feature_names or ['aqi', 'pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3']
    metrics = {}
    for i, name in enumerate(feature_names):
        if i < y_true.shape[-1]:
            true_col = y_true[..., i].flatten()
            pred_col = y_pred[..., i].flatten()
            mse = mean_squared_error(true_col, pred_col)
            metrics[f'{name}_mse'] = mse
            metrics[f'{name}_rmse'] = np.sqrt(mse)
            metrics[f'{name}_mae'] = mean_absolute_error(true_col, pred_col)
            metrics[f'{name}_r2'] = r2_score(true_col, pred_col)

    # 整体指标
    metrics['overall_mse'] = mean_squared_error(y_true.flatten(), y_pred.flatten())
    metrics['overall_rmse'] = np.sqrt(metrics['overall_mse'])
    metrics['overall_mae'] = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    metrics['overall_r2'] = r2_score(y_true.flatten(), y_pred.flatten())

    return metrics
