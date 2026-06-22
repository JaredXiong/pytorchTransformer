"""End-to-end integration test for semi-supervised VMD pipeline."""
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from air_quality.config import config
from air_quality.data.processor import AirQualityDataProcessor
from air_quality.data.vmd import VMDDecomposer
from air_quality.data.vmd_features import apply_vmd_to_aqi
from air_quality.training import SemiSupervisedTrainer


class TestFullSemiPipeline(unittest.TestCase):
    def _create_mock_excel(self, path: str) -> None:
        """创建小型 Excel 测试数据集"""
        n = 250  # 足够支持 40/40/20 划分 + 14 天窗口
        dates = pd.date_range('2020-01-01', periods=n, freq='D')
        df = pd.DataFrame({
            'pubtime': dates,
            'AQI': np.random.RandomState(42).randint(20, 200, n),
            'PM2.5': np.random.RandomState(43).randint(10, 150, n),
            'PM10': np.random.RandomState(44).randint(20, 200, n),
            'NO2': np.random.RandomState(45).randint(10, 80, n),
            'SO2': np.random.RandomState(46).randint(5, 50, n),
            'CO': np.random.RandomState(47).uniform(0.5, 3.0, n),
            'O3': np.random.RandomState(48).randint(20, 160, n),
        })
        df.to_excel(path, index=False)

    def test_semi_pipeline_runs(self):
        """端到端半监督 pipeline 应能跑通"""
        # 用 mock 数据
        with tempfile.TemporaryDirectory() as tmp:
            excel_path = os.path.join(tmp, 'mock.xlsx')
            self._create_mock_excel(excel_path)
            original_data_file = config.data.data_file
            try:
                config.update_from_dict({'data': {'data_file': excel_path}})

                processor = AirQualityDataProcessor(
                    seq_length=14,
                    prediction_days=3,
                    output_size=config.model.output_size,
                )
                data, scaler, features, dates = processor.load_and_preprocess_data(excel_path)
                X, y = processor.create_sequences(data)

                # 应用 VMD
                decomposer = VMDDecomposer(K=4)
                X_vmd = apply_vmd_to_aqi(X, decomposer)

                # 三段划分
                (X_lab, y_lab), (X_unl, y_unl), (X_te, y_te) = processor.split_three_way(X_vmd, y)

                # 验证划分
                self.assertGreater(len(X_lab), 0)
                self.assertGreater(len(X_unl), 0)
                self.assertGreater(len(X_te), 0)
                # 输入维度：6 (非 AQI 污染物) + 4 (周期性编码) + K (VMD IMF) = 10 + K
                # 这里 K=4（decomposer 实例化时指定），所以是 10+4=14
                self.assertEqual(X_vmd.shape[2], 10 + decomposer.K)

                # 跑半监督训练（epochs=2 以避开 PyTorch OneCycleLR 在 epochs=1 时的 zero-division 错误）
                trainer = SemiSupervisedTrainer(
                    model_type='vmd_cnn_bilstm_attention',
                    input_size=X_vmd.shape[2],
                    device='cpu',
                    teacher_epochs=2,
                    student_epochs=2,
                    pseudo_confidence_threshold=0.5,
                    batch_size=8,
                )
                student, metrics = trainer.fit(
                    X_labeled=X_lab, y_labeled=y_lab,
                    X_unlabeled=X_unl, y_unlabeled=y_unl,
                    X_test=X_te, y_test=y_te,
                )

                # 验证训练完成
                self.assertIsNotNone(student)
                self.assertIn('test_loss', metrics)
                self.assertFalse(np.isnan(metrics['test_loss']))
                # 验证 4 阶段都记录了损失
                self.assertGreater(len(trainer.history['teacher_losses']), 0)
                self.assertGreater(len(trainer.history['student_losses']), 0)
                self.assertGreater(len(trainer.history['pseudo_label_rate']), 0)
                # 验证 predictions 形状
                self.assertEqual(metrics['predictions'].shape, (len(X_te), 3, 7))
            finally:
                # 恢复原始配置
                config.update_from_dict({'data': {'data_file': original_data_file}})

    def test_pipeline_without_vmd(self):
        """不启用 VMD 时 pipeline 应能跑通"""
        with tempfile.TemporaryDirectory() as tmp:
            excel_path = os.path.join(tmp, 'mock.xlsx')
            self._create_mock_excel(excel_path)
            original_data_file = config.data.data_file
            try:
                config.update_from_dict({
                    'data': {'data_file': excel_path},
                    'vmd': {'enabled': False},
                })

                processor = AirQualityDataProcessor(
                    seq_length=14,
                    prediction_days=3,
                    output_size=config.model.output_size,
                )
                data, scaler, features, dates = processor.load_and_preprocess_data(excel_path)
                X, y = processor.create_sequences(data)

                # 不应用 VMD
                X_no_vmd = X  # 原始 9 维

                (X_lab, y_lab), (X_unl, y_unl), (X_te, y_te) = processor.split_three_way(X_no_vmd, y)

                trainer = SemiSupervisedTrainer(
                    model_type='vmd_cnn_bilstm_attention',
                    input_size=X_no_vmd.shape[2],  # 9
                    device='cpu',
                    teacher_epochs=2,
                    student_epochs=2,
                    pseudo_confidence_threshold=0.5,
                    batch_size=8,
                )
                student, metrics = trainer.fit(
                    X_labeled=X_lab, y_labeled=y_lab,
                    X_unlabeled=X_unl, y_unlabeled=y_unl,
                    X_test=X_te, y_test=y_te,
                )
                self.assertIsNotNone(student)
                self.assertFalse(np.isnan(metrics['test_loss']))
            finally:
                config.update_from_dict({
                    'data': {'data_file': original_data_file},
                    'vmd': {'enabled': True},
                })


if __name__ == '__main__':
    unittest.main()
