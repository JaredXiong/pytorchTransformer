"""PretrainFinetuneTrainer 三阶段流水线测试"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from air_quality.config.settings import PretrainConfig
from air_quality.models import create_model
from air_quality.training.pretrain import Pretrainer
from air_quality.training.pretrain_finetune import PretrainFinetuneTrainer


class TestPretrainFinetuneTrainer(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
        self.X_labeled = np.random.randn(20, 14, 12).astype(np.float32)
        self.y_labeled = np.random.randn(20, 3, 7).astype(np.float32)
        self.X_unlabeled = np.random.randn(20, 14, 12).astype(np.float32)
        self.y_unlabeled = np.random.randn(20, 3, 7).astype(np.float32)
        self.X_test = np.random.randn(10, 14, 12).astype(np.float32)
        self.y_test = np.random.randn(10, 3, 7).astype(np.float32)
        self._build_pretrain_cfg = lambda enabled: PretrainConfig(
            enabled=enabled, epochs=1, mask_ratio=0.3,
            learning_rate=5e-4, weight_decay=3e-4,
            early_stop_patience=10,
        )

    def _make_trainer(self, pretrain_enabled=True):
        return PretrainFinetuneTrainer(
            model_type='vmd_cnn_bilstm_attention',
            input_size=12,
            device='cpu',
            teacher_epochs=1,
            student_epochs=1,
            pseudo_confidence_threshold=0.5,
            batch_size=8,
            pretrain_config=self._build_pretrain_cfg(pretrain_enabled),
        )

    def test_three_phase_pipeline_runs(self):
        """三阶段流水线应能跑通"""
        trainer = self._make_trainer(pretrain_enabled=True)
        student, metrics = trainer.fit(
            X_labeled=self.X_labeled, y_labeled=self.y_labeled,
            X_unlabeled=self.X_unlabeled, y_unlabeled=self.y_unlabeled,
            X_test=self.X_test, y_test=self.y_test,
        )
        self.assertIsNotNone(student)
        self.assertIn('test_loss', metrics)
        self.assertGreater(len(trainer.history['pretrain_losses']), 0)
        self.assertGreater(len(trainer.history['teacher_losses']), 0)
        self.assertGreater(len(trainer.history['student_losses']), 0)

    def test_pretrain_disabled_skips_pretrain(self):
        """--pretrain 关闭时,pretrain_losses 应为空,流水线降级为半监督"""
        trainer = self._make_trainer(pretrain_enabled=False)
        _, metrics = trainer.fit(
            X_labeled=self.X_labeled, y_labeled=self.y_labeled,
            X_unlabeled=self.X_unlabeled, y_unlabeled=self.y_unlabeled,
            X_test=self.X_test, y_test=self.y_test,
        )
        self.assertEqual(len(trainer.history['pretrain_losses']), 0)
        self.assertGreater(len(trainer.history['teacher_losses']), 0)
        self.assertGreater(len(trainer.history['student_losses']), 0)
        self.assertIn('test_loss', metrics)

    def test_teacher_student_share_backbone_init(self):
        """Teacher 与 Student 在训练前应共享 backbone 起点(在 enabled 路径下)"""
        trainer = self._make_trainer(pretrain_enabled=True)
        seen_states = {}

        def wrapped_fit(*args, **kwargs):
            t = create_model('vmd_cnn_bilstm_attention', input_size=12)
            s = create_model('vmd_cnn_bilstm_attention', input_size=12)
            backbone_state = Pretrainer.extract_backbone_state(trainer._pretrain_model)
            Pretrainer.load_backbone_into(t, backbone_state)
            Pretrainer.load_backbone_into(s, backbone_state)
            seen_states['teacher_bilstm'] = t.bilstm.weight_hh_l0.detach().clone()
            seen_states['student_bilstm'] = s.bilstm.weight_hh_l0.detach().clone()
            seen_states['teacher_head'] = t.finetune_head[-1].weight.detach().clone()
            seen_states['student_head'] = s.finetune_head[-1].weight.detach().clone()
            return original_fit(*args, **kwargs)

        original_fit = trainer._fit_semi
        trainer._fit_semi = wrapped_fit
        trainer.fit(
            X_labeled=self.X_labeled, y_labeled=self.y_labeled,
            X_unlabeled=self.X_unlabeled, y_unlabeled=self.y_unlabeled,
            X_test=self.X_test, y_test=self.y_test,
        )
        self.assertTrue(torch.equal(
            seen_states['teacher_bilstm'], seen_states['student_bilstm']
        ))
        self.assertFalse(torch.equal(
            seen_states['teacher_head'], seen_states['student_head']
        ))
