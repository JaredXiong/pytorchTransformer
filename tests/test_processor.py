"""Tests for data processor after input/target split."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from air_quality.config import config
from air_quality.data import AirQualityDataProcessor


class TestProcessorSplit(unittest.TestCase):
    def setUp(self):
        self.processor = AirQualityDataProcessor(
            seq_length=14,
            prediction_days=3,
        )

    def test_scaler_is_7d(self):
        """Scaler fitted on 7 pollutant features only."""
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
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            df.to_excel(f.name, index=False)
            path = f.name
        try:
            data, scaler, features, dates = self.processor.load_and_preprocess_data(path)
            # scaler fitted on 7 pollutants
            self.assertEqual(scaler.n_features_in_, 7)
            # full data array is 9-dim
            self.assertEqual(data.shape[1], 9)
        finally:
            os.unlink(path)

    def test_create_sequences_split_shapes(self):
        """X has 9 features (last 2 are month/season), y has 7 features (pollutants only)."""
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
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            df.to_excel(f.name, index=False)
            path = f.name
        try:
            data, scaler, features, dates = self.processor.load_and_preprocess_data(path)
            X, y = self.processor.create_sequences(data)
            # X contains month/season
            self.assertEqual(X.shape[-1], 9)
            # y contains pollutants only
            self.assertEqual(y.shape[-1], 7)
        finally:
            os.unlink(path)


class TestValidatePrediction(unittest.TestCase):
    def test_clamps_aqi_and_co_7dim(self):
        from air_quality.data.processor import validate_prediction
        # AQI at idx 0 should clamp to 500; CO at idx 5 should clamp to 50
        pred = np.array([[1000.0, 50.0, 80.0, 40.0, 20.0, 100.0, 60.0]])  # AQI=1000, CO=100
        out = validate_prediction(pred)
        self.assertEqual(out[0, 0], 500.0)
        self.assertEqual(out[0, 5], 50.0)

    def test_no_month_season_validation(self):
        """Function should not reference month/season indices (which don't exist anymore)."""
        from air_quality.data import processor as proc_module
        import inspect
        src = inspect.getsource(proc_module.validate_prediction)
        self.assertNotIn('month', src)
        self.assertNotIn('season', src)


if __name__ == '__main__':
    unittest.main()
