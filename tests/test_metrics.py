"""Tests for metrics module — 7-feature outputs only."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from air_quality.data import calculate_metrics
from air_quality.training import MetricsTracker


class TestMetricsSeven(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.y_true = np.random.randn(10, 3, 7)
        self.y_pred = self.y_true + np.random.randn(10, 3, 7) * 0.1

    def test_metrics_default_no_month_season(self):
        """Default MetricsTracker has 7 feature names, no month/season."""
        mt = MetricsTracker()
        self.assertEqual(len(mt.feature_names), 7)
        self.assertNotIn('month', mt.feature_names)
        self.assertNotIn('season', mt.feature_names)

    def test_calculate_metrics_7_features(self):
        """calculate_metrics returns metrics only for 7 features."""
        m = calculate_metrics(self.y_true, self.y_pred)
        for feature in ['aqi', 'pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3']:
            self.assertIn(f'{feature}_mse', m)
        # No month/season metrics
        self.assertNotIn('month_mse', m)
        self.assertNotIn('season_mse', m)


if __name__ == '__main__':
    unittest.main()
