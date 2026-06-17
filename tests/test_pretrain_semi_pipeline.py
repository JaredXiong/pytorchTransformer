"""端到端预训练-微调半监督 pipeline 集成测试"""
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from air_quality.config import config
from air_quality.config.settings import PretrainConfig
from air_quality.data.processor import AirQualityDataProcessor
from air_quality.data.vmd import VMDDecomposer
from air_quality.data.vmd_features import apply_vmd_to_aqi
from air_quality.training import PretrainFinetuneTrainer


class TestPretrainSemiPipeline(unittest.TestCase):
    """端到端预训练+半监督 pipeline 集成测试"""

    def _create_mock_excel(self, path: str) -> None:
        """构造小型 Excel mock 数据集"""
        n = 250
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

    def test_pretrain_semi_pipeline_runs(self):
        """端到端预训练+半监督 pipeline 应能跑通"""
        with tempfile.TemporaryDirectory() as tmp:
            excel_path = os.path.join(tmp, 'mock.xlsx')
            self._create_mock_excel(excel_path)

            # 临时切换数据文件
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

                self.assertGreater(len(X_lab), 0)
                self.assertGreater(len(X_unl), 0)
                self.assertGreater(len(X_te), 0)
                # 输入维度：8 (非 AQI) + 4 (VMD K) = 12
                self.assertEqual(X_vmd.shape[2], 12)

                # 启用预训练配置
                pretrain_cfg = PretrainConfig(
                    enabled=True,
                    epochs=1,
                    mask_ratio=0.3,
                    learning_rate=5e-4,
                )

                # 跑三阶段流水线(epochs=2 以避开 OneCycleLR zero-division)
                trainer = PretrainFinetuneTrainer(
                    model_type='vmd_cnn_bilstm_attention',
                    input_size=X_vmd.shape[2],
                    device='cpu',
                    teacher_epochs=2,
                    student_epochs=2,
                    pseudo_confidence_threshold=0.5,
                    batch_size=8,
                    pretrain_config=pretrain_cfg,
                )
                student, metrics = trainer.fit(
                    X_labeled=X_lab, y_labeled=y_lab,
                    X_unlabeled=X_unl, y_unlabeled=y_unl,
                    X_test=X_te, y_test=y_te,
                )

                # 验证三阶段全部完成
                self.assertIsNotNone(student)
                self.assertIn('test_loss', metrics)
                self.assertFalse(np.isnan(metrics['test_loss']))
                # 验证三阶段 history 都已记录
                self.assertGreater(len(trainer.history['pretrain_losses']), 0)
                self.assertGreater(len(trainer.history['teacher_losses']), 0)
                self.assertGreater(len(trainer.history['student_losses']), 0)
                # 验证 predictions 形状
                self.assertEqual(metrics['predictions'].shape, (len(X_te), 3, 7))
            finally:
                # 恢复原始配置
                config.update_from_dict({'data': {'data_file': original_data_file}})

    def test_pretrain_disabled_works(self):
        """禁用预训练时(pretrain_config.enabled=False)pipeline 应能跑通"""
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

                decomposer = VMDDecomposer(K=4)
                X_vmd = apply_vmd_to_aqi(X, decomposer)
                (X_lab, y_lab), (X_unl, y_unl), (X_te, y_te) = processor.split_three_way(X_vmd, y)

                # 显式禁用预训练
                pretrain_cfg = PretrainConfig(enabled=False)

                trainer = PretrainFinetuneTrainer(
                    model_type='vmd_cnn_bilstm_attention',
                    input_size=X_vmd.shape[2],
                    device='cpu',
                    teacher_epochs=2,
                    student_epochs=2,
                    pseudo_confidence_threshold=0.5,
                    batch_size=8,
                    pretrain_config=pretrain_cfg,
                )
                student, metrics = trainer.fit(
                    X_labeled=X_lab, y_labeled=y_lab,
                    X_unlabeled=X_unl, y_unlabeled=y_unl,
                    X_test=X_te, y_test=y_te,
                )

                self.assertIsNotNone(student)
                self.assertFalse(np.isnan(metrics['test_loss']))
                # 禁用预训练时 pretrain_losses 应为空
                self.assertEqual(len(trainer.history['pretrain_losses']), 0)
                # 但 teacher/student 仍应有记录
                self.assertGreater(len(trainer.history['teacher_losses']), 0)
                self.assertGreater(len(trainer.history['student_losses']), 0)
            finally:
                config.update_from_dict({'data': {'data_file': original_data_file}})


if __name__ == '__main__':
    unittest.main()
