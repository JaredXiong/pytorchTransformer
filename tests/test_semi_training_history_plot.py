"""Regression tests for plot_training_history with semi-supervised history.

Bug: SemiSupervisedTrainer.history uses keys ``teacher_losses`` /
``student_losses`` / ``pretrain_losses`` / ``pseudo_label_rate`` while
``plot_training_history`` only handled ``train_loss`` / ``test_loss`` /
``lr``. The result was a blank figure (no curves drawn). This test pins
the new multi-mode behaviour.
"""
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from air_quality.visualization import plot_training_history


class TestPlotTrainingHistorySemi(unittest.TestCase):
    """plot_training_history must render non-blank curves for semi history."""

    def _assert_png_has_content(self, path: str) -> None:
        """A blank figure saved by Agg is ~5 KB; with content it's > 10 KB."""
        self.assertTrue(os.path.exists(path), f"plot not saved: {path}")
        size_kb = os.path.getsize(path) / 1024
        self.assertGreater(
            size_kb, 10,
            f"PNG suspiciously small ({size_kb:.1f} KB) — figure may be blank: {path}",
        )

    def test_semi_history_with_teacher_student_renders_curves(self):
        history = {
            'teacher_losses': [0.5, 0.4, 0.35, 0.32, 0.30],
            'student_losses': [0.4, 0.3, 0.25, 0.22, 0.20, 0.18, 0.17],
            'pseudo_label_rate': [0.6, 0.75],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'semi_history.png')
            plot_training_history(history, save_path=out)
            self._assert_png_has_content(out)

    def test_pretrain_semi_history_renders_three_phases(self):
        history = {
            'pretrain_losses': [0.8, 0.6, 0.5],
            'teacher_losses': [0.4, 0.35, 0.30],
            'student_losses': [0.3, 0.25, 0.20, 0.18],
            'pseudo_label_rate': [0.5, 0.7, 0.85],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'pretrain_history.png')
            plot_training_history(history, save_path=out)
            self._assert_png_has_content(out)

    def test_legacy_supervised_history_still_renders(self):
        """The old train_loss/test_loss/lr path must keep working."""
        history = {
            'train_loss': [0.5, 0.4, 0.3, 0.2],
            'test_loss': [0.55, 0.45, 0.35, 0.25],
            'lr': [1e-3, 5e-4, 1e-4, 1e-5],
            'best_epoch': 3,
            'best_loss': 0.25,
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'supervised_history.png')
            plot_training_history(history, save_path=out)
            self._assert_png_has_content(out)

    def test_empty_history_does_not_crash(self):
        """Empty / unknown history should not raise — just produce a blank canvas."""
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'empty.png')
            plot_training_history({}, save_path=out)  # must not raise
            self.assertTrue(os.path.exists(out))


if __name__ == '__main__':
    unittest.main()
