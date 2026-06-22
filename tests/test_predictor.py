"""Tests for predictor's backtest() and forecast() modes."""
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from air_quality.config import config
from air_quality.data import AirQualityDataProcessor
from air_quality.models import create_model
from air_quality.inference import AirQualityPredictor


def _train_and_save_model(model, processor, tmpdir):
    """Train a tiny model and save weights + scaler so predictor can load them."""
    import joblib
    n = 60
    df = pd.DataFrame({
        'pubtime': pd.date_range('2024-01-01', periods=n, freq='D'),
        'aqi': np.random.rand(n) * 100,
        'pm2_5_24h': np.random.rand(n) * 50,
        'pm10_24h': np.random.rand(n) * 80,
        'no2_24h': np.random.rand(n) * 40,
        'so2_24h': np.random.rand(n) * 20,
        'co_24h': np.random.rand(n) * 5,
        'o3_8h_24h': np.random.rand(n) * 60,
    })
    excel = os.path.join(tmpdir, 'data.xlsx')
    df.to_excel(excel, index=False)

    data, scaler, features, dates = processor.load_and_preprocess_data(excel)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    x = torch.FloatTensor(data[:14]).unsqueeze(0)
    y = torch.FloatTensor(data[14:17, :7]).unsqueeze(0)  # 7-dim target
    for _ in range(20):
        out = model(x)[:, -3:, :]
        loss = criterion(out, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    weights = os.path.join(tmpdir, 'model.pth')
    scaler_pkl = os.path.join(tmpdir, 'scaler.pkl')
    # checkpoint 必须包含 input_size / output_size（与 save_artifacts 格式一致）
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model.__class__.__name__,
        'input_size': model.input_size,
        'output_size': model.output_size,
    }, weights)
    joblib.dump(scaler, scaler_pkl)
    return weights, scaler_pkl, data, dates


class TestForecast(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

    def test_forecast_input_shape_and_value_ranges(self):
        """forecast accepts (14, 11) raw input and returns (3, 7) predictions."""
        with tempfile.TemporaryDirectory() as tmp:
            processor = AirQualityDataProcessor(seq_length=14, prediction_days=3)
            model = create_model('lstm', input_size=11, output_size=7)
            weights, scaler_pkl, _, _ = _train_and_save_model(model, processor, tmp)

            p = AirQualityPredictor(model_weights_path=weights, scaler_path=scaler_pkl)
            p.load_model()

            # 构造 (14, 11) 输入：7 污染物 + 4 周期性编码特征
            # month=6 (sin=0, cos=-1), season=2 (sin=1, cos=0)
            input_seq = np.array([
                [50.0, 30.0, 60.0, 25.0, 10.0, 1.0, 50.0, 0.0, -1.0, 1.0, 0.0],
                [55.0, 32.0, 62.0, 26.0, 11.0, 1.1, 52.0, 0.0, -1.0, 1.0, 0.0],
            ] * 7, dtype=float)  # (14, 11)

            result = p.forecast(input_seq)
            self.assertEqual(result['predictions'].shape, (3, 7))

    def test_forecast_rejects_invalid_month(self):
        with tempfile.TemporaryDirectory() as tmp:
            processor = AirQualityDataProcessor(seq_length=14, prediction_days=3)
            model = create_model('lstm', input_size=11, output_size=7)
            weights, scaler_pkl, _, _ = _train_and_save_model(model, processor, tmp)
            p = AirQualityPredictor(model_weights_path=weights, scaler_path=scaler_pkl)
            p.load_model()

            bad = np.zeros((14, 11))
            # 周期性编码特征的范围是 [-1, 1]，设置一个超出范围的值
            bad[:, 7] = 2.0  # invalid month_sin (超出 [-1, 1] 范围)
            with self.assertRaises(ValueError):
                p.forecast(bad)


class TestBacktest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

    def test_backtest_returns_predictions_for_each_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            processor = AirQualityDataProcessor(seq_length=14, prediction_days=3)
            model = create_model('lstm', input_size=11, output_size=7)
            weights, scaler_pkl, data, dates = _train_and_save_model(model, processor, tmp)

            p = AirQualityPredictor(model_weights_path=weights, scaler_path=scaler_pkl)
            p.load_model()

            result = p.backtest(data=data, dates=dates, prediction_days=3)
            n = len(data)
            expected_windows = n - 14 - 3 + 1
            self.assertEqual(result['y_pred'].shape, (expected_windows, 3, 7))
            self.assertEqual(result['y_true'].shape, (expected_windows, 3, 7))
            self.assertIn('metrics', result)


if __name__ == '__main__':
    unittest.main()
