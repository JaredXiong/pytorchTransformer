"""Tests for config module."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from air_quality.config import config


class TestCalendarConfig(unittest.TestCase):
    def test_input_size_unchanged(self):
        """Input includes 7 pollutants + 4 cyclic calendar features = 11 features."""
        self.assertEqual(config.model.input_size, 11)

    def test_output_size_added(self):
        """Output is 7 pollutants only."""
        self.assertEqual(config.model.output_size, 7)

    def test_target_columns_count(self):
        """7 target columns."""
        self.assertEqual(len(config.data.target_columns), 7)

    def test_target_columns_no_calendar(self):
        """target_columns does NOT contain month/season."""
        self.assertNotIn('month', config.data.target_columns)
        self.assertNotIn('season', config.data.target_columns)


if __name__ == '__main__':
    unittest.main()
