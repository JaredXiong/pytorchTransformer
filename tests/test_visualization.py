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

        with tempfile.TemporaryDirectory() as tmp:
            save_path = os.path.join(tmp, 'backtest.png')
            result = plot_backtest_results(
                y_true=y_true, y_pred=y_pred, dates=dates,
                save_path=save_path,
            )
            self.assertEqual(result, save_path)
            self.assertTrue(os.path.exists(save_path))
            self.assertGreater(os.path.getsize(save_path), 1000)


if __name__ == '__main__':
    unittest.main()
