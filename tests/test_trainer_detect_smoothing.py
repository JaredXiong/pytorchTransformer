"""Tests for ModelTrainer detect_over_smoothing 集成。"""
import os
import sys
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from air_quality.models import create_model
from air_quality.training import ModelTrainer


class TestTrainerDetectSmoothing(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.B = 4
        self.T = 14
        self.F = 7
        # 构造足以过 train 1-2 epoch 的小数据集
        self.x = torch.randn(32, self.T, 9)
        self.y = torch.randn(32, 3, self.F)
        self.x_test = torch.randn(16, self.T, 9)
        self.y_test = torch.randn(16, 3, self.F)

    def _make_loaders(self):
        train_loader = DataLoader(TensorDataset(self.x, self.y), batch_size=8, shuffle=True)
        test_loader = DataLoader(TensorDataset(self.x_test, self.y_test), batch_size=8, shuffle=False)
        return train_loader, test_loader

    def test_history_records_smoothing_flags(self):
        model = create_model('lstm', input_size=9, output_size=7)
        train_loader, test_loader = self._make_loaders()
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device='cpu',
            best_model_path=None,
            loss_type='mse',
            detect_smoothing=True,
            smoothing_threshold=0.1,
        )
        trainer.train(num_epochs=2)
        flags = trainer.history.get('smoothing_flags_per_epoch', [])
        self.assertEqual(len(flags), 2, '每 epoch 应记录一次 smoothing flags')
        for entry in flags:
            self.assertIn('n_flagged', entry)
            self.assertIn('flags', entry)

    def test_disable_detect_smoothing_still_works(self):
        """关闭 detect_smoothing 时不应报错，history 中 flags 仍记录但恒为 0。"""
        model = create_model('lstm', input_size=9, output_size=7)
        train_loader, test_loader = self._make_loaders()
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device='cpu',
            best_model_path=None,
            loss_type='mse',
            detect_smoothing=False,
        )
        trainer.train(num_epochs=1)
        flags = trainer.history.get('smoothing_flags_per_epoch', [])
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0]['n_flagged'], 0)

    def test_loss_kwargs_passed_to_factory(self):
        """mse_antismooth 路径下 AntiSmoothingLoss 应在 epoch 循环中接受 set_epoch。"""
        model = create_model('lstm', input_size=9, output_size=7)
        train_loader, test_loader = self._make_loaders()
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device='cpu',
            best_model_path=None,
            loss_type='mse_antismooth',
            loss_kwargs={'lambda_var': 0.1, 'lambda_diff': 0.05,
                         'tau_var': 0.5, 'tau_diff': 0.5,
                         'lambda_warmup_epochs': 5},
            detect_smoothing=False,
        )
        # AntiSmoothingLoss 内部应能接受 set_epoch
        from air_quality.training.losses import AntiSmoothingLoss
        self.assertIsInstance(trainer.criterion, AntiSmoothingLoss)
        trainer.train(num_epochs=2)
        # 训练完成后 criterion.current_epoch 应已被 set_epoch 更新
        self.assertGreaterEqual(trainer.criterion.current_epoch, 1)


if __name__ == '__main__':
    unittest.main()
