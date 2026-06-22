"""Tests for VMD and semi-supervised configuration classes."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from air_quality.config import config


class TestSemiSupervisedConfig(unittest.TestCase):
    def test_vmd_config_defaults(self):
        self.assertTrue(config.vmd.enabled)
        self.assertEqual(config.vmd.K, 3)
        self.assertEqual(config.vmd.alpha, 2000)

    def test_semi_supervised_defaults(self):
        self.assertFalse(config.semi.enabled)
        self.assertEqual(config.semi.labeled_ratio, 0.4)
        self.assertEqual(config.semi.unlabeled_ratio, 0.4)
        self.assertEqual(config.semi.test_ratio, 0.2)
        self.assertEqual(config.semi.teacher_epochs, 80)
        self.assertEqual(config.semi.student_epochs, 120)
        self.assertEqual(config.semi.pseudo_confidence_threshold, 0.85)
        self.assertEqual(config.semi.pseudo_loss_weight, 0.5)

    def test_config_update_from_dict(self):
        """config.update_from_dict 应能更新嵌套字段"""
        original_vmd_k = config.vmd.K
        original_semi_enabled = config.semi.enabled
        try:
            config.update_from_dict({
                'semi': {'enabled': True, 'labeled_ratio': 0.5},
                'vmd': {'K': 6},
            })
            self.assertTrue(config.semi.enabled)
            self.assertEqual(config.semi.labeled_ratio, 0.5)
            self.assertEqual(config.vmd.K, 6)
        finally:
            # 恢复默认值
            config.update_from_dict({
                'semi': {'enabled': original_semi_enabled, 'labeled_ratio': 0.4},
                'vmd': {'K': original_vmd_k},
            })

    def test_semi_ratios_sum_to_one(self):
        """labeled + unlabeled + test 比例之和应等于 1.0"""
        total = config.semi.labeled_ratio + config.semi.unlabeled_ratio + config.semi.test_ratio
        self.assertAlmostEqual(total, 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
