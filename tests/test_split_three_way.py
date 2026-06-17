"""Tests for split_three_way chronological data splitting."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from air_quality.data.processor import AirQualityDataProcessor


class TestSplitThreeWay(unittest.TestCase):
    def setUp(self):
        self.processor = AirQualityDataProcessor()
        np.random.seed(0)
        self.X = np.random.randn(100, 14, 9)
        self.y = np.random.randn(100, 3, 7)

    def test_default_ratios(self):
        """默认 0.4/0.4/0.2 划分"""
        (Xl, yl), (Xu, yu), (Xt, yt) = self.processor.split_three_way(self.X, self.y)
        self.assertEqual(len(Xl), 40)
        self.assertEqual(len(Xu), 40)
        self.assertEqual(len(Xt), 20)
        self.assertEqual(len(yl), 40)
        self.assertEqual(len(yu), 40)
        self.assertEqual(len(yt), 20)

    def test_custom_ratios(self):
        """自定义 0.5/0.3/0.2 划分"""
        (Xl, yl), (Xu, yu), (Xt, yt) = self.processor.split_three_way(
            self.X, self.y, ratios=(0.5, 0.3, 0.2)
        )
        self.assertEqual(len(Xl), 50)
        self.assertEqual(len(Xu), 30)
        self.assertEqual(len(Xt), 20)

    def test_chronological_order(self):
        """必须按时间顺序划分，不打乱"""
        (Xl, _), (Xu, _), (Xt, _) = self.processor.split_three_way(
            self.X, self.y, ratios=(0.4, 0.4, 0.2)
        )
        # labeled 段的所有元素应在前 40 个位置
        # unlabeled 段的所有元素应在中间 40 个位置
        # test 段的所有元素应在最后 20 个位置
        # 通过比较原始索引位置间接验证：labeled 段的最大索引 < unlabeled 段的最小索引
        # 由于数据是 random 的（np.random.randn），我们通过 shape 推断索引顺序
        # 检查每段元素总和的连续性即可：
        # labeled 末尾 + 1 = unlabeled 起始
        self.assertEqual(len(Xl) + len(Xu) + len(Xt), 100)
        # 用切片验证：labeled 段等于原始 X 的前 40 个
        np.testing.assert_array_equal(Xl, self.X[:40])
        np.testing.assert_array_equal(Xu, self.X[40:80])
        np.testing.assert_array_equal(Xt, self.X[80:])

    def test_no_overlap(self):
        """三段之间无重叠"""
        (Xl, _), (Xu, _), (Xt, _) = self.processor.split_three_way(
            self.X, self.y, ratios=(0.4, 0.4, 0.2)
        )
        total = len(Xl) + len(Xu) + len(Xt)
        self.assertEqual(total, len(self.X))

    def test_invalid_ratios(self):
        """比例之和不等于 1 应报错"""
        with self.assertRaises(AssertionError):
            self.processor.split_three_way(self.X, self.y, ratios=(0.5, 0.3, 0.3))
