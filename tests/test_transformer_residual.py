"""Tests for TransformerModel anti-smoothing architecture changes."""
import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from air_quality.models import create_model
from air_quality.models.transformer import TransformerModel


class TestTransformerResidual(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.batch = 2
        self.seq_len = 14
        self.input_size = 9
        self.output_size = 7

    def test_residual_weight_and_feature_scale_exist(self):
        model = create_model('transformer', input_size=self.input_size, output_size=self.output_size)
        self.assertTrue(hasattr(model, 'residual_weight'), 'TransformerModel must have residual_weight')
        self.assertTrue(hasattr(model, 'feature_scale'), 'TransformerModel must have feature_scale')
        self.assertIsInstance(model.residual_weight, nn.Parameter)
        self.assertIsInstance(model.feature_scale, nn.Parameter)
        # residual_weight 是 1 维标量（logit）；feature_scale 是 (output_size,)
        self.assertEqual(model.residual_weight.shape, torch.Size([]))
        self.assertEqual(model.feature_scale.shape, torch.Size([self.output_size]))

    def test_residual_in_state_dict(self):
        model = create_model('transformer', input_size=self.input_size, output_size=self.output_size)
        state = model.state_dict()
        self.assertIn('residual_weight', state)
        self.assertIn('feature_scale', state)
        self.assertIn('input_proj.weight', state)
        self.assertIn('residual_proj.weight', state)

    def test_forward_shape_unchanged(self):
        model = create_model('transformer', input_size=self.input_size, output_size=self.output_size)
        x = torch.randn(self.batch, self.seq_len, self.input_size)
        y = model(x)
        self.assertEqual(y.shape, (self.batch, self.seq_len, self.output_size))

    def test_sigmoid_residual_weight_in_range(self):
        """sigmoid(residual_weight) 应始终 ∈ (0, 1)，与初值无关。"""
        model = TransformerModel(
            input_size=self.input_size, output_size=self.output_size,
            residual_weight=10.0,  # 极端高初值
        )
        with torch.no_grad():
            w = torch.sigmoid(model.residual_weight)
        self.assertGreater(float(w), 0.0)
        self.assertLess(float(w), 1.0)
        # 默认 0.3 的初值：sigmoid(logit(0.3)) ≈ 0.574
        default_model = TransformerModel(input_size=self.input_size, output_size=self.output_size)
        with torch.no_grad():
            w0 = torch.sigmoid(default_model.residual_weight)
        self.assertGreater(float(w0), 0.4)
        self.assertLess(float(w0), 0.8)

    def test_gradients_flow_through_residual_path(self):
        model = create_model('transformer', input_size=self.input_size, output_size=self.output_size)
        model.train()
        x = torch.randn(self.batch, self.seq_len, self.input_size, requires_grad=False)
        y_target = torch.randn(self.batch, self.seq_len, self.output_size)
        y_pred = model(x)
        loss = (y_pred - y_target).pow(2).mean()
        loss.backward()
        # 残差 / feature_scale / input_proj 都有梯度
        self.assertIsNotNone(model.residual_weight.grad)
        self.assertIsNotNone(model.feature_scale.grad)
        self.assertIsNotNone(model.input_proj.weight.grad)
        self.assertIsNotNone(model.residual_proj.weight.grad)
        # 残差权重与 feature_scale 的梯度都应是有限值
        self.assertTrue(torch.isfinite(model.residual_weight.grad).all().item())
        self.assertTrue(torch.isfinite(model.feature_scale.grad).all().item())

    def test_anti_smoothing_does_not_break_other_models(self):
        """Transformer 重构不应影响其他模型。"""
        from air_quality.models import create_model as _cm
        for mt in ('lstm', 'cnn', 'hybrid'):
            with self.subTest(model=mt):
                m = _cm(mt, input_size=self.input_size, output_size=self.output_size)
                x = torch.randn(self.batch, self.seq_len, self.input_size)
                y = m(x)
                self.assertEqual(y.shape, (self.batch, self.seq_len, self.output_size))


if __name__ == '__main__':
    unittest.main()
