"""PretrainConfig 单元测试"""
import os
import sys
import unittest

# 让 unittest 可以独立运行此文件
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), '..', 'src')
)

from air_quality.config import config


class TestPretrainConfig(unittest.TestCase):
    def setUp(self):
        self._orig_enabled = config.pretrain.enabled
        self._orig_epochs = config.pretrain.epochs
        self._orig_mask_ratio = config.pretrain.mask_ratio

    def tearDown(self):
        config.update_from_dict({
            'pretrain': {
                'enabled': self._orig_enabled,
                'epochs': self._orig_epochs,
                'mask_ratio': self._orig_mask_ratio,
            }
        })

    def test_defaults(self):
        """默认值"""
        self.assertFalse(config.pretrain.enabled)
        self.assertEqual(config.pretrain.epochs, 60)
        self.assertEqual(config.pretrain.mask_ratio, 0.3)
        self.assertEqual(config.pretrain.learning_rate, 5e-4)
        self.assertEqual(config.pretrain.target, 'imf')

    def test_update_from_dict(self):
        """update_from_dict 应能更新 pretrain 字段"""
        config.update_from_dict({
            'pretrain': {
                'enabled': True,
                'epochs': 20,
                'mask_ratio': 0.5,
            }
        })
        self.assertTrue(config.pretrain.enabled)
        self.assertEqual(config.pretrain.epochs, 20)
        self.assertEqual(config.pretrain.mask_ratio, 0.5)
