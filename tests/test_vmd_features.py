"""Tests for VMD-AQI feature transformation utilities."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from air_quality.data.vmd_features import apply_vmd_to_aqi
from air_quality.data.vmd import VMDDecomposer


class TestVMDFeatures(unittest.TestCase):
    def test_apply_vmd_to_aqi_shape(self):
        """应返回 (N, 14, 8+K)"""
        decomposer = VMDDecomposer(K=4)
        X = np.random.randn(10, 14, 9)
        result = apply_vmd_to_aqi(X, decomposer)
        self.assertEqual(result.shape, (10, 14, 8 + 4))

    def test_apply_vmd_to_aqi_K2(self):
        """K=2 时维度应为 8+2=10"""
        decomposer = VMDDecomposer(K=2)
        X = np.random.randn(3, 14, 9)
        result = apply_vmd_to_aqi(X, decomposer)
        self.assertEqual(result.shape, (3, 14, 10))

    def test_apply_vmd_to_aqi_K6(self):
        """K=6 时维度应为 8+6=14"""
        decomposer = VMDDecomposer(K=6)
        X = np.random.randn(3, 14, 9)
        result = apply_vmd_to_aqi(X, decomposer)
        self.assertEqual(result.shape, (3, 14, 14))

    def test_apply_vmd_to_aqi_replaces_aqi_column(self):
        """AQI 列（第 0 维）应被替换为 VMD 分量，不再是原始 AQI 值"""
        decomposer = VMDDecomposer(K=4)
        X = np.random.RandomState(42).randn(3, 14, 9)
        result = apply_vmd_to_aqi(X, decomposer)
        # VMD 分解的非线性特性意味着 IMF 分量与原始 AQI 不相同
        # 但仅检查不是完全一致即可（如果代码错误导致 AQI 仍保留，
        # 此断言会失败）
        self.assertFalse(np.allclose(result[:, :, 0], X[:, :, 0]))


if __name__ == '__main__':
    unittest.main()
