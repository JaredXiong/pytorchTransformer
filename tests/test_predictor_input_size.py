"""Regression test for predictor.load_model() with non-default input_size.

Bug reproduction: When VMD is enabled during semi-supervised training, the
model is trained with input_size = 8 + K (e.g. 12 for K=4). The training
script used to save the checkpoint without recording input_size. As a
result, predictor.load_model() fell back to config.model.input_size
(default 9) and failed with a size mismatch when loading the
VMD-CNN-BiLSTM-Attention weights.
"""
import os
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from air_quality.config import config
from air_quality.data import AirQualityDataProcessor
from air_quality.models import create_model
from air_quality.inference import AirQualityPredictor


class TestPredictorInputSize(unittest.TestCase):
    """Verify that predictor.load_model() honours the saved input_size."""

    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

    def _save_vmd_model(self, tmpdir: str, input_size: int = 12):
        """Save a VMD-CNN-BiLSTM-Attention checkpoint with input_size=12.

        Mirrors the post-condition of VMD semi-supervised training where
        apply_vmd_to_aqi produces 12-dim features (K=4 IMF + 8 others).
        Uses default cnn_filters / bilstm_hidden / bilstm_layers / dropout
        (matching what scripts/train.py does for the VMD model).
        """
        weights = os.path.join(tmpdir, 'model.pth')
        scaler_pkl = os.path.join(tmpdir, 'scaler.pkl')

        model = create_model(
            'vmd_cnn_bilstm_attention',
            input_size=input_size,
            output_size=7,
        )
        # Run a single forward pass to materialise the model parameters.
        x = torch.randn(1, 14, input_size)
        _ = model(x)

        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'model_type': model.__class__.__name__,
                'input_size': model.input_size,
                'output_size': model.output_size,
                'vmd_params': {
                    'K': getattr(model, 'vmd_K', 4),
                    'alpha': 2000, 'tau': 0, 'DC': 0, 'init': 1, 'tol': 1e-7,
                } if hasattr(model, 'vmd_K') else None,
            },
            weights,
        )

        # Minimal scaler (7 pollutant dims only, like the real pipeline).
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(np.random.rand(30, 7))
        joblib.dump(scaler, scaler_pkl)
        return weights, scaler_pkl

    def test_load_model_with_vmd_checkpoint_does_not_raise(self):
        """Loading a 14-dim VMD checkpoint should not raise a size mismatch."""
        with tempfile.TemporaryDirectory() as tmp:
            weights, scaler_pkl = self._save_vmd_model(tmp, input_size=14)
            predictor = AirQualityPredictor(
                model_weights_path=weights, scaler_path=scaler_pkl,
            )
            # The bug: this raised ValueError because the predictor
            # instantiated the model with input_size=9.
            try:
                predictor.load_model()
            except ValueError as e:
                self.fail(
                    f"predictor.load_model() raised ValueError for a "
                    f"14-dim VMD checkpoint: {e}"
                )
            self.assertIsNotNone(predictor.model)
            # Sanity check: the loaded model expects 14 input features.
            self.assertEqual(predictor.model.input_size, 14)

    def test_save_artifacts_records_input_size(self):
        """save_artifacts should embed input_size and vmd_params in the checkpoint dict."""
        from scripts.train import save_artifacts

        with tempfile.TemporaryDirectory() as tmp:
            # Redirect checkpoint/scaler paths to tmp.
            weights = os.path.join(tmp, 'model.pth')
            scaler_pkl = os.path.join(tmp, 'scaler.pkl')
            original_weights = config.file.model_save_path
            original_scaler = config.file.scaler_path
            try:
                config.update_from_dict({
                    'file': {
                        'model_save_path': weights,
                        'scaler_path': scaler_pkl,
                    }
                })

                model = create_model(
                    'vmd_cnn_bilstm_attention',
                    input_size=14, output_size=7,
                )
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaler.fit(np.random.rand(30, 7))

                save_artifacts(model, scaler)

                ckpt = torch.load(weights, map_location='cpu')
                self.assertEqual(ckpt.get('input_size'), 14)
                self.assertEqual(ckpt.get('output_size'), 7)
                # VMD 模型应记录完整 vmd_params（含 K），供 load_model 直接构造
                self.assertEqual(ckpt.get('vmd_params', {}).get('K'), 4)
            finally:
                config.update_from_dict({
                    'file': {
                        'model_save_path': original_weights,
                        'scaler_path': original_scaler,
                    }
                })

    def test_forecast_with_vmd_model_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights, scaler_pkl = self._save_vmd_model(tmp, input_size=14)
            predictor = AirQualityPredictor(
                model_weights_path=weights, scaler_path=scaler_pkl,
            )
            predictor.load_model()
            # predictor 应已构建 VMDDecomposer
            self.assertIsNotNone(predictor._vmd_decomposer)
            self.assertEqual(predictor._vmd_decomposer.K, 4)

            # 构造 14 天 11 维 raw 输入（VMD 模型下 forecaster 内部会做 VMD 分解）
            # 7 污染物 + 4 周期性编码特征 (month_sin, month_cos, season_sin, season_cos)
            # month=6 → sin=0, cos=-1; season=2 → sin=1, cos=0
            input_seq = np.array([
                [50.0, 30.0, 60.0, 25.0, 10.0, 1.0, 50.0, 0.0, -1.0, 1.0, 0.0],
                [55.0, 32.0, 62.0, 26.0, 11.0, 1.1, 52.0, 0.0, -1.0, 1.0, 0.0],
                [60.0, 35.0, 65.0, 28.0, 12.0, 1.2, 55.0, 0.0, -1.0, 1.0, 0.0],
                [58.0, 33.0, 64.0, 27.0, 11.5, 1.15, 53.0, 0.0, -1.0, 1.0, 0.0],
                [56.0, 31.0, 63.0, 26.5, 11.0, 1.1, 52.0, 0.0, -1.0, 1.0, 0.0],
                [54.0, 30.0, 62.0, 26.0, 10.5, 1.05, 51.0, 0.0, -1.0, 1.0, 0.0],
                [52.0, 29.0, 61.0, 25.5, 10.0, 1.0, 50.0, 0.0, -1.0, 1.0, 0.0],
                [50.0, 28.0, 60.0, 25.0, 9.5, 0.95, 49.0, 0.0, -1.0, 1.0, 0.0],
                [48.0, 27.0, 59.0, 24.5, 9.0, 0.9, 48.0, 0.0, -1.0, 1.0, 0.0],
                [46.0, 26.0, 58.0, 24.0, 8.5, 0.85, 47.0, 0.0, -1.0, 1.0, 0.0],
                [44.0, 25.0, 57.0, 23.5, 8.0, 0.8, 46.0, 0.0, -1.0, 1.0, 0.0],
                [42.0, 24.0, 56.0, 23.0, 7.5, 0.75, 45.0, 0.0, -1.0, 1.0, 0.0],
                [40.0, 23.0, 55.0, 22.5, 7.0, 0.7, 44.0, 0.0, -1.0, 1.0, 0.0],
                [38.0, 22.0, 54.0, 22.0, 6.5, 0.65, 43.0, 0.0, -1.0, 1.0, 0.0],
            ], dtype=float)

            result = predictor.forecast(input_seq)
            self.assertEqual(result['predictions'].shape, (3, 7))


if __name__ == '__main__':
    unittest.main()
