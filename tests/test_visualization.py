"""Tests for plot_backtest_results."""
import os
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use('Agg')  # non-interactive
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from air_quality.visualization import plot_backtest_results


class TestPlotBacktest(unittest.TestCase):
    def test_creates_figure_and_saves(self):
        n = 30
        y_true = np.random.rand(n, 3, 7)
        y_pred = y_true + np.random.randn(n, 3, 7) * 0.1
        dates = [['2024-01-{:02d}'.format(i % 28 + 1)] * 3 for i in range(n)]
        feature_names = ['AQI', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = plot_backtest_results(
                y_true=y_true, y_pred=y_pred, dates=dates,
                save_path=tmp, model_type='test_model',
            )
            self.assertTrue(os.path.isdir(out_dir))
            for name in feature_names:
                fname = name.replace('.', '_') + '.png'
                fpath = os.path.join(out_dir, fname)
                self.assertTrue(os.path.exists(fpath), f'{fname} not found')
                self.assertGreater(os.path.getsize(fpath), 1000)

    def test_y_true_raw_overrides_y_true(self):
        """当提供 y_true_raw 时，图表应使用原始未裁剪的真值，避免 y 轴被强行截断。"""
        n = 10
        prediction_days = 3
        # y_true 是被裁剪到 AQI<=500、CO<=50 的值
        y_true = np.full((n, prediction_days, 7), 100.0)
        # y_true_raw 是原始未裁剪的真值：AQI=600（>500），CO=80（>50），PM2.5=900
        y_true_raw = y_true.copy()
        y_true_raw[..., 0] = 600.0   # AQI 超过 500
        y_true_raw[..., 1] = 900.0   # PM2.5 实际范围可达 1000
        y_true_raw[..., 5] = 80.0    # CO 超过 50

        y_pred = y_true.copy()  # 预测值被裁剪，与 y_true 范围一致
        dates = [['2024-01-{:02d}'.format(i % 28 + 1)] * prediction_days for i in range(n)]

        with tempfile.TemporaryDirectory() as tmp:
            out_dir_raw = plot_backtest_results(
                y_true=y_true, y_pred=y_pred, dates=dates,
                save_path=tmp, model_type='with_raw',
                y_true_raw=y_true_raw,
            )
            self.assertTrue(os.path.isdir(out_dir_raw))
            for name in ['AQI', 'PM2_5', 'CO']:
                fname = name.replace('.', '_') + '.png'
                fpath = os.path.join(out_dir_raw, fname)
                self.assertTrue(os.path.exists(fpath), f'{fname} not found')
                self.assertGreater(os.path.getsize(fpath), 1000)

    def test_backward_compatible_without_y_true_raw(self):
        """不传 y_true_raw 时应回退到 y_true（旧行为）。"""
        n = 10
        prediction_days = 3
        y_true = np.full((n, prediction_days, 7), 100.0)
        y_pred = y_true.copy()
        dates = [['2024-01-{:02d}'.format(i % 28 + 1)] * prediction_days for i in range(n)]

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = plot_backtest_results(
                y_true=y_true, y_pred=y_pred, dates=dates,
                save_path=tmp, model_type='no_raw',
            )
            self.assertTrue(os.path.isdir(out_dir))
            for name in ['AQI', 'PM2_5', 'CO']:
                fname = name.replace('.', '_') + '.png'
                fpath = os.path.join(out_dir, fname)
                self.assertTrue(os.path.exists(fpath), f'{fname} not found')


if __name__ == '__main__':
    unittest.main()
