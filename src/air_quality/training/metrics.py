"""
评估指标模块
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict


class MetricsTracker:
    """多指标追踪器"""

    def __init__(self, feature_names: list = None):
        self.feature_names = feature_names or [
            'aqi', 'pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3'
        ]
        self.history = {
            'train_loss': [],
            'test_loss': [],
            'metrics': []
        }

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算评估指标"""
        metrics = {}

        # 整体指标
        metrics['overall_mse'] = mean_squared_error(y_true.flatten(), y_pred.flatten())
        metrics['overall_rmse'] = np.sqrt(metrics['overall_mse'])
        metrics['overall_mae'] = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        metrics['overall_r2'] = r2_score(y_true.flatten(), y_pred.flatten())

        # 分特征指标
        for i, name in enumerate(self.feature_names):
            if i < y_true.shape[-1]:
                true_col = y_true[..., i].flatten()
                pred_col = y_pred[..., i].flatten()
                mse = mean_squared_error(true_col, pred_col)
                metrics[f'{name}_mse'] = mse
                metrics[f'{name}_rmse'] = np.sqrt(mse)
                metrics[f'{name}_mae'] = mean_absolute_error(true_col, pred_col)
                metrics[f'{name}_r2'] = r2_score(true_col, pred_col)

        return metrics

    def detect_over_smoothing(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.1) -> Dict[str, bool]:
        """检测过度平滑现象

        过度平滑的特征：
        1. 预测值方差远小于真实值方差
        2. 预测值变化幅度远小于真实值
        """
        results = {}

        for i, name in enumerate(self.feature_names[:7]):  # 只检查污染物特征
            if i < y_true.shape[-1]:
                true_std = np.std(y_true[..., i])
                pred_std = np.std(y_pred[..., i])

                # 方差比
                std_ratio = pred_std / (true_std + 1e-8)

                # 变化幅度比
                true_diff = np.diff(y_true[..., i]).std()
                pred_diff = np.diff(y_pred[..., i]).std()
                diff_ratio = pred_diff / (true_diff + 1e-8)

                # 检测结果
                results[name] = std_ratio < threshold or diff_ratio < threshold

        return results

    def update(self, train_loss: float, test_loss: float, metrics: Dict[str, float]):
        """更新历史记录"""
        self.history['train_loss'].append(train_loss)
        self.history['test_loss'].append(test_loss)
        self.history['metrics'].append(metrics)

    def get_best_metrics(self) -> Dict[str, float]:
        """获取最佳指标"""
        if not self.history['metrics']:
            return {}
        # 找到 RMSE 最低的时刻
        rmse_values = [m['overall_rmse'] for m in self.history['metrics']]
        best_idx = np.argmin(rmse_values)
        return self.history['metrics'][best_idx]
