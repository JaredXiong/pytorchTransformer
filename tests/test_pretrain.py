"""Pretrainer 单元测试"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from air_quality.models.vmd_cnn_bilstm_attention import (
    VMDCNNBILSTMAttentionModel,
)
from air_quality.training.pretrain import Pretrainer, compute_masked_mse


class TestMaskedMSE(unittest.TestCase):
    def test_returns_finite_scalar(self):
        """应返回有限标量"""
        pred = np.random.randn(4, 14, 4).astype(np.float32)
        target = np.random.randn(4, 14, 4).astype(np.float32)
        mask = np.zeros((4, 14, 4), dtype=np.float32)
        mask[0, 3, 1] = 1.0
        loss = compute_masked_mse(pred, target, mask)
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(loss, 0.0)

    def test_only_masked_positions_contribute(self):
        """loss 应仅在 mask=1 的位置计算"""
        pred = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
        target = np.array([[[0.0, 0.0], [0.0, 0.0]]], dtype=np.float32)
        mask_zero = np.zeros_like(pred)
        mask_one = np.ones_like(pred)
        loss_zero = compute_masked_mse(pred, target, mask_zero)
        loss_one = compute_masked_mse(pred, target, mask_one)
        self.assertAlmostEqual(loss_zero, 0.0, places=6)
        self.assertGreater(loss_one, 0.0)


class TestPretrainerMaskGeneration(unittest.TestCase):
    def test_mask_shape(self):
        """生成 (B, T, K) 的 0/1 掩码"""
        mask = Pretrainer.generate_mask((8, 14, 4), ratio=0.3, seed=0)
        self.assertEqual(mask.shape, (8, 14, 4))
        self.assertTrue(set(np.unique(mask)).issubset({0.0, 1.0}))

    def test_mask_ratio_approximate(self):
        """遮盖比例应接近 ratio"""
        np.random.seed(42)
        mask = Pretrainer.generate_mask((100, 14, 4), ratio=0.3)
        actual_ratio = 1.0 - mask.mean()
        self.assertAlmostEqual(actual_ratio, 0.3, delta=0.05)

    def test_mask_reproducible_with_seed(self):
        """相同 seed 应得到相同掩码"""
        m1 = Pretrainer.generate_mask((5, 14, 4), ratio=0.3, seed=123)
        m2 = Pretrainer.generate_mask((5, 14, 4), ratio=0.3, seed=123)
        np.testing.assert_array_equal(m1, m2)


class TestPretrainerFit(unittest.TestCase):
    def setUp(self):
        self.model = VMDCNNBILSTMAttentionModel(
            input_size=12, output_size=7, vmd_K=4,
            cnn_filters=16, bilstm_hidden=16, bilstm_layers=1, dropout=0.1,
        )
        self.pretrainer = Pretrainer(
            model=self.model, device='cpu',
            learning_rate=5e-4, weight_decay=3e-4,
            mask_ratio=0.3, early_stop_patience=10,
        )
        np.random.seed(0)
        torch.manual_seed(0)
        self.X = np.random.randn(16, 14, 12).astype(np.float32)

    def test_fit_runs_and_returns_history(self):
        """fit 应能跑通并返回 history"""
        history = self.pretrainer.fit(self.X, epochs=2, batch_size=4)
        self.assertIn('train_losses', history)
        self.assertGreater(len(history['train_losses']), 0)

    def test_fit_loss_decreases_or_stable(self):
        """最后 N 步的损失应不大于前 N 步的最大值(允许 30% 波动)"""
        history = self.pretrainer.fit(self.X, epochs=4, batch_size=4)
        losses = history['train_losses']
        half = len(losses) // 2
        first_max = max(losses[:half])
        second_max = max(losses[half:])
        self.assertLessEqual(second_max, first_max * 1.3)

    def test_extract_backbone_state(self):
        """extract_backbone_state 应丢弃 pretrain_head"""
        self.pretrainer.fit(self.X, epochs=1, batch_size=4)
        state = Pretrainer.extract_backbone_state(self.model)
        bad = [k for k in state if 'pretrain_head' in k]
        self.assertEqual(bad, [])
        good = [k for k in state if any(
            k.startswith(prefix) for prefix in
            ['input_proj', 'conv3', 'conv5', 'conv7', 'bn', 'bilstm', 'attention']
        )]
        self.assertGreater(len(good), 0)

    def test_gradient_flows_through_backbone(self):
        """backbone 参数应有非零梯度"""
        model = VMDCNNBILSTMAttentionModel(
            input_size=12, output_size=7, vmd_K=4,
            cnn_filters=16, bilstm_hidden=16, bilstm_layers=1, dropout=0.0,
        )
        pretrainer = Pretrainer(
            model=model, device='cpu', mask_ratio=0.3,
        )
        x = torch.FloatTensor(np.random.randn(4, 14, 12))
        mask = torch.FloatTensor(
            Pretrainer.generate_mask((4, 14, 4), ratio=0.3, seed=0)
        )
        out = model(x, mode='pretrain')
        target = x[:, :, :4]
        loss = ((out - target) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)
        loss.backward()
        self.assertIsNotNone(model.bilstm.weight_hh_l0.grad)
        self.assertIsNotNone(model.pretrain_head.weight.grad)
        self.assertFalse(torch.all(model.bilstm.weight_hh_l0.grad == 0))
        self.assertFalse(torch.all(model.pretrain_head.weight.grad == 0))


if __name__ == '__main__':
    unittest.main()
