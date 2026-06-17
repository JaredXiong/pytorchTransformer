"""Tests for SemiSupervisedTrainer and pseudo-label confidence."""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from air_quality.training.semi_supervised import (
    SemiSupervisedTrainer,
    compute_pseudo_confidence,
)


class TestPseudoConfidence(unittest.TestCase):
    def test_confidence_shape(self):
        """置信度输出形状应为 (N,)"""
        pred = np.random.randn(10, 3, 7)
        conf = compute_pseudo_confidence(pred)
        self.assertEqual(conf.shape, (10,))
        self.assertTrue((conf >= 0).all() and (conf <= 1).all())

    def test_low_variance_high_confidence(self):
        """低方差预测应有高置信度"""
        # 全部相同值 → std=0 → 置信度=1.0
        pred = np.zeros((5, 3, 7))
        conf = compute_pseudo_confidence(pred)
        np.testing.assert_array_almost_equal(conf, np.ones(5))

    def test_high_variance_low_confidence(self):
        """高方差预测应有低置信度"""
        np.random.seed(0)
        pred = np.random.randn(5, 3, 7) * 100
        conf = compute_pseudo_confidence(pred)
        self.assertTrue((conf < 0.1).all())


class TestSemiSupervisedTrainer(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
        # 构造小数据集
        self.X_labeled = np.random.randn(20, 14, 12).astype(np.float32)  # 8+K=12
        self.y_labeled = np.random.randn(20, 3, 7).astype(np.float32)
        self.X_unlabeled = np.random.randn(20, 14, 12).astype(np.float32)
        self.y_unlabeled = np.random.randn(20, 3, 7).astype(np.float32)
        self.X_test = np.random.randn(10, 14, 12).astype(np.float32)
        self.y_test = np.random.randn(10, 3, 7).astype(np.float32)

    def test_train_runs_4_phases(self):
        """应能完成 4 阶段训练"""
        trainer = SemiSupervisedTrainer(
            model_type='vmd_cnn_bilstm_attention',
            input_size=12,
            device='cpu',
            teacher_epochs=2,
            student_epochs=2,
            pseudo_confidence_threshold=0.5,
        )
        student, metrics = trainer.fit(
            X_labeled=self.X_labeled,
            y_labeled=self.y_labeled,
            X_unlabeled=self.X_unlabeled,
            y_unlabeled=self.y_unlabeled,
            X_test=self.X_test,
            y_test=self.y_test,
        )
        self.assertIsNotNone(student)
        self.assertIn('test_loss', metrics)
        self.assertGreater(len(trainer.history['teacher_losses']), 0)
        self.assertGreater(len(trainer.history['student_losses']), 0)
        self.assertGreater(len(trainer.history['pseudo_label_rate']), 0)

    def test_pseudo_label_rate_recorded(self):
        """伪标签保留率应被记录在 history 中"""
        trainer = SemiSupervisedTrainer(
            model_type='vmd_cnn_bilstm_attention',
            input_size=12,
            device='cpu',
            teacher_epochs=1,
            student_epochs=1,
            pseudo_confidence_threshold=0.5,
        )
        trainer.fit(
            X_labeled=self.X_labeled,
            y_labeled=self.y_labeled,
            X_unlabeled=self.X_unlabeled,
            y_unlabeled=self.y_unlabeled,
            X_test=self.X_test,
            y_test=self.y_test,
        )
        # 伪标签保留率应在 [0, 1] 之间
        rate = trainer.history['pseudo_label_rate'][0]
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)


if __name__ == '__main__':
    unittest.main()