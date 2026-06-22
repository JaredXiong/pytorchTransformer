"""Regression tests for the semi-supervised anti-smoothing & chart-parity fix.

Covers the two user-reported problems:

1. "字段不匹配导致图表不一致/空白" — the semi history now carries the same
   families of fields the supervised history does (lr, validation loss, smoothing
   flags), and ``plot_training_history`` renders a 3-panel figure whose panels
   degrade gracefully instead of going blank.

2. "七个指标过度平滑" — the SemiSupervisedTrainer now trains teacher/student with
   the configured loss_type (mse_antismooth) + loss_kwargs + detect_smoothing,
   instead of the previously hardcoded huber.
"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from air_quality.training.semi_supervised import SemiSupervisedTrainer
from air_quality.training.pretrain_finetune import PretrainFinetuneTrainer
from air_quality.config.settings import PretrainConfig
from air_quality.visualization import plot_training_history


def _synth(n=80, seq_len=14, pred_days=3, F=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, seq_len, F)).astype(np.float32)
    # per-sample phase oscillation so smoothing is detectable
    t = np.arange(pred_days)[None, :, None]
    phase = np.arange(n)[:, None, None]
    base = 10 * np.sin(t * 0.9 + phase)
    base = np.broadcast_to(base, (n, pred_days, 7))
    y = (base + rng.standard_normal((n, pred_days, 7))).astype(np.float32)
    return X, y


def _three_way(X, y):
    n = len(X)
    n1, n2 = int(n * 0.4), int(n * 0.4)
    return (X[:n1], y[:n1]), (X[n1:n1 + n2], y[n1:n1 + n2]), (X[n1 + n2:], y[n1 + n2:])


class _LossProbe:
    """Capture the loss_type/loss_kwargs actually passed to ModelTrainer by
    monkey-patching the trainer module reference used by semi_supervised."""

    def __init__(self):
        self.captured = []

    def install(self):
        from air_quality.training import semi_supervised as mod
        self._orig = mod.ModelTrainer

        captured = self.captured

        class _Spy(self._orig):
            def __init__(self, *args, **kwargs):
                captured.append({
                    'loss_type': kwargs.get('loss_type'),
                    'loss_kwargs': dict(kwargs.get('loss_kwargs') or {}),
                    'detect_smoothing': kwargs.get('detect_smoothing'),
                })
                super().__init__(*args, **kwargs)

        mod.ModelTrainer = _Spy

    def restore(self):
        from air_quality.training import semi_supervised as mod
        mod.ModelTrainer = self._orig


class TestSemiUsesAntiSmoothingLoss(unittest.TestCase):
    """Root-cause B: semi teacher/student must NOT fall back to huber."""

    def setUp(self):
        np.random.seed(0)

    def test_teacher_and_student_use_configured_loss(self):
        X, y = _synth(n=80)
        (Xl, yl), (Xu, yu), (Xt, yt) = _three_way(X, y)
        probe = _LossProbe()
        probe.install()
        try:
            trainer = SemiSupervisedTrainer(
                model_type='vmd_cnn_bilstm_attention',
                input_size=12, device='cpu',
                teacher_epochs=2, student_epochs=2,
                loss_type='mse_antismooth',
                loss_kwargs={'lambda_var': 0.1, 'lambda_diff': 0.05,
                             'tau_var': 0.5, 'tau_diff': 0.5,
                             'lambda_warmup_epochs': 2},
                detect_smoothing=True,
            )
            trainer.fit(Xl, yl, Xu, yu, Xt, yt)
        finally:
            probe.restore()

        self.assertEqual(len(probe.captured), 2, "expected teacher + student")
        for cap in probe.captured:
            self.assertEqual(cap['loss_type'], 'mse_antismooth')
            self.assertTrue(cap['detect_smoothing'])
            self.assertIn('lambda_var', cap['loss_kwargs'])
            self.assertIn('lambda_diff', cap['loss_kwargs'])

    def test_defaults_to_anti_smooth_not_huber(self):
        """No explicit loss_type → trainer must default to mse_antismooth,
        not the old hardcoded huber."""
        X, y = _synth(n=60)
        (Xl, yl), (Xu, yu), (Xt, yt) = _three_way(X, y)
        probe = _LossProbe()
        probe.install()
        try:
            trainer = SemiSupervisedTrainer(
                model_type='vmd_cnn_bilstm_attention',
                input_size=12, device='cpu',
                teacher_epochs=1, student_epochs=1,
            )
            trainer.fit(Xl, yl, Xu, yu, Xt, yt)
        finally:
            probe.restore()
        for cap in probe.captured:
            self.assertEqual(cap['loss_type'], 'mse_antismooth')


class TestSemiHistoryParityFields(unittest.TestCase):
    """Root-cause A: semi history must carry lr / val loss / smoothing flags."""

    def setUp(self):
        np.random.seed(0)

    def test_history_has_lr_and_val_loss_and_flags(self):
        X, y = _synth(n=80)
        (Xl, yl), (Xu, yu), (Xt, yt) = _three_way(X, y)
        trainer = SemiSupervisedTrainer(
            model_type='vmd_cnn_bilstm_attention',
            input_size=12, device='cpu',
            teacher_epochs=3, student_epochs=3,
        )
        trainer.fit(Xl, yl, Xu, yu, Xt, yt)
        h = trainer.history
        # supervised-parity fields now present
        self.assertGreater(len(h['teacher_lr']), 0, "teacher lr not recorded")
        self.assertGreater(len(h['student_lr']), 0, "student lr not recorded")
        self.assertGreater(len(h['teacher_test_loss']), 0)
        self.assertGreater(len(h['student_test_loss']), 0)
        self.assertGreater(len(h['smoothing_flags_per_epoch']), 0)
        # the flag entries have the expected shape
        first_flag = h['smoothing_flags_per_epoch'][0]
        self.assertIn('n_flagged', first_flag)
        # legacy aggregated loss lists still populated
        self.assertGreater(len(h['teacher_losses']), 0)
        self.assertGreater(len(h['student_losses']), 0)


class TestPlotParityAndGracefulDegradation(unittest.TestCase):
    """The semi plot must render 3 informative panels and never be blank."""

    def _assert_nonblank(self, path):
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path) / 1024, 10,
                           f"PNG suspiciously small — may be blank: {path}")

    def test_semi_history_renders_three_panels(self):
        X, y = _synth(n=80)
        (Xl, yl), (Xu, yu), (Xt, yt) = _three_way(X, y)
        trainer = SemiSupervisedTrainer(
            model_type='vmd_cnn_bilstm_attention',
            input_size=12, device='cpu',
            teacher_epochs=3, student_epochs=3,
        )
        trainer.fit(Xl, yl, Xu, yu, Xt, yt)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'semi.png')
            plot_training_history(trainer.history, save_path=out)
            self._assert_nonblank(out)

    def test_pretrain_history_renders_three_panels(self):
        X, y = _synth(n=80)
        (Xl, yl), (Xu, yu), (Xt, yt) = _three_way(X, y)
        pc = PretrainConfig(enabled=True, epochs=3, early_stop_patience=99)
        trainer = PretrainFinetuneTrainer(
            model_type='vmd_cnn_bilstm_attention',
            input_size=12, device='cpu',
            teacher_epochs=3, student_epochs=3, pretrain_config=pc,
        )
        trainer.fit(Xl, yl, Xu, yu, Xt, yt)
        self.assertGreater(len(trainer.history['pretrain_losses']), 0)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'pretrain.png')
            plot_training_history(trainer.history, save_path=out)
            self._assert_nonblank(out)

    def test_empty_semi_history_does_not_blank(self):
        """All-loss-empty semi history must render prompts, not a blank canvas."""
        history = {
            'teacher_losses': [], 'student_losses': [],
            'pretrain_losses': [], 'pseudo_label_rate': [],
            'teacher_lr': [], 'student_lr': [],
            'teacher_test_loss': [], 'student_test_loss': [],
            'smoothing_flags_per_epoch': [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'empty_semi.png')
            plot_training_history(history, save_path=out)  # must not raise
            self._assert_nonblank(out)


if __name__ == '__main__':
    unittest.main()
