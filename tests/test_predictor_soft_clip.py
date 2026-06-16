"""Tests for predictor soft_clip 反平滑推理。"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from air_quality.inference.predictor import _soft_saturate, HARD_MAX_PER_FEATURE


class TestSoftSaturate(unittest.TestCase):
    def test_no_excess_returns_input(self):
        arr = np.array([[100, 200, 50, 30, 20, 5, 100]], dtype=np.float64)
        out = _soft_saturate(arr, HARD_MAX_PER_FEATURE, scale=20.0)
        # 所有值都在上限内，输出应几乎等于输入
        np.testing.assert_allclose(out, arr, atol=1e-6)

    def test_excess_soft_clips(self):
        """AQI=1000（excess=500）应被软饱和：输出 < 500，但仍是有限值。"""
        arr = np.array([[1000.0, 50, 80, 40, 20, 5, 60]], dtype=np.float64)
        out = _soft_saturate(arr, HARD_MAX_PER_FEATURE, scale=20.0)
        # AQI 索引 0：上限 500，excess=500，应软饱和到约 500 - 20*log1p(25) ≈ 500 - 65.78 ≈ 434
        self.assertLess(out[0, 0], 500.0)
        self.assertGreater(out[0, 0], 0.0)
        self.assertTrue(np.isfinite(out[0, 0]))
        # CO 索引 5：上限 50，输入 5，无 excess → 输出 = 5
        np.testing.assert_allclose(out[0, 5], 5.0, atol=1e-6)
        # 其他特征：上限 inf → 原样
        np.testing.assert_allclose(out[0, 1:], arr[0, 1:], atol=1e-6)

    def test_infinite_max_passes_through(self):
        arr = np.array([[1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9]], dtype=np.float64)
        out = _soft_saturate(arr, HARD_MAX_PER_FEATURE, scale=20.0)
        # AQI 索引 0：excess 极大，应被软饱和但仍是有限值
        self.assertTrue(np.isfinite(out[0, 0]))
        # CO 索引 5：上限 50，excess 极大
        self.assertTrue(np.isfinite(out[0, 5]))
        # 其他索引：上限 inf → 原样
        self.assertEqual(out[0, 1], 1e9)
        self.assertEqual(out[0, 2], 1e9)
        self.assertEqual(out[0, 3], 1e9)
        self.assertEqual(out[0, 4], 1e9)
        self.assertEqual(out[0, 6], 1e9)

    def test_dimension_mismatch_falls_back_to_hard_clip(self):
        """维度不匹配时退化到硬裁剪以保证数值稳定。"""
        # arr 4 列，max 3 列 → shape[-1] != shape[0] 触发硬裁剪回退
        arr = np.array([[600, 60, 80, 40]], dtype=np.float64)
        max_arr = np.array([500, 50, np.inf])  # 3 features only
        out = _soft_saturate(arr, max_arr, scale=20.0)
        # AQI=600 应被硬裁剪到 500
        self.assertEqual(out[0, 0], 500.0)
        # CO=60 应被硬裁剪到 50
        self.assertEqual(out[0, 1], 50.0)
        # 第三个无上限 → 原样
        self.assertEqual(out[0, 2], 80.0)

    def test_smaller_scale_clips_more_aggressively(self):
        """scale 越小，曲线越陡 → 输出更接近 max。"""
        arr = np.array([[1000.0]], dtype=np.float64)
        max_arr = np.array([500.0])
        out_large = _soft_saturate(arr, max_arr, scale=100.0)
        out_small = _soft_saturate(arr, max_arr, scale=5.0)
        # scale=5 时输出应更接近 500（更陡）
        self.assertGreater(out_small[0, 0], out_large[0, 0])
        self.assertLess(out_small[0, 0], 500.0)
        self.assertLess(out_large[0, 0], 500.0)


if __name__ == '__main__':
    unittest.main()
