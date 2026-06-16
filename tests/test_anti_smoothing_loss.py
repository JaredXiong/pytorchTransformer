"""Tests for AntiSmoothingLoss."""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from air_quality.training.losses import AntiSmoothingLoss, get_loss_function


class TestAntiSmoothingLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.B = 8
        self.T = 3  # prediction_days
        self.F = 7
        # 构造有明显方差的 target
        t = torch.linspace(0, 1, self.T)
        self.target = torch.stack([torch.stack([t + i * 0.1] * self.F) for i in range(self.B)])  # (B,T,F)

    def test_loss_is_scalar(self):
        loss_fn = AntiSmoothingLoss()
        pred = self.target + torch.randn_like(self.target) * 0.1
        loss = loss_fn(pred, self.target)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_loss_finite(self):
        loss_fn = AntiSmoothingLoss()
        pred = self.target + torch.randn_like(self.target) * 0.5
        loss = loss_fn(pred, self.target)
        self.assertTrue(torch.isfinite(loss).item())

    def test_loss_decreases_when_pred_matches_target_better(self):
        """当 pred 的 std/diff 接近 target（而非过度平滑）时，loss 应更小。

        构造：
        - target：有真实结构（递增）
        - flat_pred：常量（std=0），应受反平滑惩罚
        - good_pred：target + 微小噪声（std/diff 接近 target），惩罚很小

        注意：MSE 项 flat_pred 可能也很小（接近 target 均值），
        但反平滑惩罚项会让 flat_loss 总体更大。
        """
        loss_fn = AntiSmoothingLoss()
        loss_fn.set_epoch(100)  # warmup 完成，惩罚项生效

        # flat_pred：常量（std=0），diff=0 → 应同时触发 var 和 diff 惩罚
        flat_pred = torch.full_like(self.target, self.target.mean().item())
        flat_loss = loss_fn(flat_pred, self.target)

        # good_pred：std/diff 接近 target → 几乎不触发惩罚
        good_pred = self.target + torch.randn_like(self.target) * 0.01
        good_loss = loss_fn(good_pred, self.target)

        # flat_loss 应包含额外惩罚项，good_loss 不应包含
        # 严格断言：flat_loss > good_loss
        self.assertGreater(
            float(flat_loss), float(good_loss),
            f'flat_loss={flat_loss} should be > good_loss={good_loss} '
            f'(flat_pred 触发反平滑惩罚)'
        )

    def test_warmup_zero_at_epoch_zero(self):
        loss_fn = AntiSmoothingLoss(warmup_epochs=10)
        loss_fn.set_epoch(0)
        # 纯 MSE：构造一个 variance 大但 MSE 极小的 pred，
        # 如果 warmup=0 惩罚项应被关闭，loss 接近纯 MSE。
        constant = torch.full_like(self.target, 0.5)  # MSE 接近 0
        large_var = torch.randn_like(self.target) * 5  # std 远大于 target
        # 此场景下 var penalty 应该贡献大，但如果 warmup=0 应当不影响
        loss_const = loss_fn(constant, self.target)
        loss_var = loss_fn(large_var, self.target)
        # 当 warmup=0 时，两个 loss 应只由 MSE 决定
        # loss_const 的 MSE 应当 < loss_var
        self.assertLess(float(loss_const), float(loss_var))

    def test_warmup_one_after_warmup_epochs(self):
        loss_fn = AntiSmoothingLoss(warmup_epochs=5)
        loss_fn.set_epoch(10)  # 远超 warmup_epochs
        flat_pred = self.target.mean(dim=1, keepdim=True).expand_as(self.target)
        # 此时惩罚项应生效
        self.assertEqual(loss_fn._warmup_factor(), 1.0)

    def test_gradient_flows(self):
        loss_fn = AntiSmoothingLoss()
        pred = (self.target + torch.randn_like(self.target) * 0.1).requires_grad_(True)
        loss = loss_fn(pred, self.target)
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all().item())

    def test_get_loss_function_dispatches_to_anti_smoothing(self):
        loss_fn = get_loss_function('mse_antismooth', lambda_var=0.1, lambda_diff=0.05)
        self.assertIsInstance(loss_fn, AntiSmoothingLoss)

    def test_get_loss_function_legacy_paths(self):
        for name in ('mse', 'huber', 'mae', 'smooth_l1'):
            with self.subTest(loss=name):
                loss_fn = get_loss_function(name)
                self.assertFalse(isinstance(loss_fn, AntiSmoothingLoss))

    def test_get_loss_function_unknown_raises(self):
        with self.assertRaises(ValueError):
            get_loss_function('not_a_loss')

    def test_short_time_axis_skips_diff(self):
        """当 T=1 时（无差分意义），应不抛错。"""
        loss_fn = AntiSmoothingLoss()
        target = torch.randn(self.B, 1, self.F)
        pred = target + torch.randn_like(target) * 0.1
        loss = loss_fn(pred, target)
        self.assertTrue(torch.isfinite(loss).item())


if __name__ == '__main__':
    unittest.main()
