"""Tests for model output shapes after output_size change."""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from air_quality.models import create_model

MODEL_TYPES = ['transformer', 'lstm', 'cnn', 'hybrid']

class TestModelOutputShape(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.batch = 2
        self.seq_len = 14
        self.input_size = 9   # 7 pollutants + month + season
        self.output_size = 7  # pollutants only

    def test_all_models_output_7(self):
        for mt in MODEL_TYPES:
            with self.subTest(model=mt):
                model = create_model(mt, input_size=self.input_size, output_size=self.output_size)
                x = torch.randn(self.batch, self.seq_len, self.input_size)
                y = model(x)
                self.assertEqual(y.shape, (self.batch, self.seq_len, self.output_size),
                                 f"{mt} expected ({self.batch},{self.seq_len},{self.output_size}), got {tuple(y.shape)}")

    def test_default_output_size(self):
        """create_model without output_size kwarg still produces 7-dim output."""
        model = create_model('hybrid', input_size=9)
        x = torch.randn(1, 14, 9)
        y = model(x)
        self.assertEqual(y.shape[-1], 7)


if __name__ == '__main__':
    unittest.main()
