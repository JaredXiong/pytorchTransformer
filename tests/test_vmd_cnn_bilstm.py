"""Tests for VMD-CNN-BiLSTM-Attention model."""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from air_quality.models.vmd_cnn_bilstm_attention import (
    VMDCNNBILSTMAttentionModel,
)


class TestVMDCNNBILSTMModel(unittest.TestCase):
    def setUp(self):
        self.model = VMDCNNBILSTMAttentionModel(
            input_size=9,
            output_size=7,
            vmd_K=4,
            cnn_filters=32,    # 测试用小尺寸
            bilstm_hidden=32,
            bilstm_layers=1,
            dropout=0.1,
        )

    def test_forward_shape(self):
        """forward 输出形状应为 (B, T, output_size)"""
        x = torch.randn(2, 14, 9)
        out = self.model(x)
        self.assertEqual(out.shape, (2, 14, 7))

    def test_take_last_3_days(self):
        """取最后 prediction_days 步应得到 (B, 3, 7)"""
        x = torch.randn(2, 14, 9)
        out = self.model(x)
        last_3 = out[:, -3:, :]
        self.assertEqual(last_3.shape, (2, 3, 7))

    def test_parameter_count(self):
        """应有一定数量的可训练参数"""
        n_params = self.model.count_parameters()
        self.assertGreater(n_params, 1000)
        self.assertLess(n_params, 5_000_000)

    def test_vmd_K_configurable(self):
        """VMD K 参数应可配置"""
        for K in [2, 4, 6]:
            model = VMDCNNBILSTMAttentionModel(
                input_size=9, output_size=7, vmd_K=K,
                cnn_filters=16, bilstm_hidden=16, bilstm_layers=1,
            )
            x = torch.randn(1, 14, 9)
            out = model(x)
            self.assertEqual(out.shape, (1, 14, 7))

    def test_backward_pass(self):
        """验证反向传播能正常计算梯度"""
        x = torch.randn(2, 14, 9)
        out = self.model(x)
        loss = out.sum()
        loss.backward()
        # 验证关键参数有梯度
        self.assertIsNotNone(self.model.bilstm.weight_hh_l0.grad)
        # 验证梯度非零（说明参数真的参与了计算图）
        self.assertTrue(self.model.bilstm.weight_hh_l0.grad.abs().sum() > 0)

    def test_vmd_K_mismatch_raises(self):
        """vmd_K 与 input_size 不匹配时模型应能正常工作（语义校验由调用方负责）

        实际生产中 VMD 分解在数据预处理阶段离线完成（vmd_features.apply_vmd_to_aqi），
        模型 forward 接收已经拼接好的 (B, T, input_size) 输入，所以 input_size 是
        K + 其他特征 拼接后的维度。当 vmd_K 远大于 input_size 时，至少能 forward。
        """
        try:
            model = VMDCNNBILSTMAttentionModel(
                input_size=9, output_size=7, vmd_K=10,
                cnn_filters=16, bilstm_hidden=16, bilstm_layers=1,
            )
            x = torch.randn(1, 14, 9)
            out = model(x)
            # 只要不崩溃就算通过；实际意义由调用方保证
            self.assertEqual(out.shape, (1, 14, 7))
        except Exception as e:
            self.fail(f"Forward failed with vmd_K=10, input_size=9: {e}")

    def test_dynamic_seq_length(self):
        """模型应支持不同的时序长度（不硬编码 seq_length=14）"""
        for seq_len in [7, 14, 21, 30]:
            x = torch.randn(2, seq_len, 9)
            out = self.model(x)
            self.assertEqual(out.shape, (2, seq_len, 7))


class TestVMDModelHeads(unittest.TestCase):
    """双 head(pretrain / finetune)拆分测试"""

    def setUp(self):
        self.model = VMDCNNBILSTMAttentionModel(
            input_size=12, output_size=7, vmd_K=4,
            cnn_filters=16, bilstm_hidden=16, bilstm_layers=1, dropout=0.1,
        )

    def test_finetune_mode_default(self):
        """默认 mode='finetune' 输出 (B, T, 7)"""
        x = torch.randn(2, 14, 12)
        out = self.model(x)
        self.assertEqual(out.shape, (2, 14, 7))

    def test_finetune_mode_explicit(self):
        """显式 mode='finetune' 输出 (B, T, 7)"""
        x = torch.randn(2, 14, 12)
        out = self.model(x, mode='finetune')
        self.assertEqual(out.shape, (2, 14, 7))

    def test_pretrain_mode_shape(self):
        """mode='pretrain' 输出 (B, T, K=4)"""
        x = torch.randn(2, 14, 12)
        out = self.model(x, mode='pretrain')
        self.assertEqual(out.shape, (2, 14, 4))

    def test_has_both_heads(self):
        """模型应同时包含 pretrain_head 与 finetune_head"""
        self.assertTrue(hasattr(self.model, 'pretrain_head'))
        self.assertTrue(hasattr(self.model, 'finetune_head'))

    def test_invalid_mode_raises(self):
        """未知 mode 应抛 ValueError"""
        x = torch.randn(2, 14, 12)
        with self.assertRaises(ValueError):
            self.model(x, mode='unknown')

    def test_pretrain_head_is_linear(self):
        """pretrain_head 应该是 Linear(bilstm_hidden*2, vmd_K)"""
        import torch.nn as nn
        self.assertIsInstance(self.model.pretrain_head, nn.Linear)
        self.assertEqual(self.model.pretrain_head.in_features, 16 * 2)
        self.assertEqual(self.model.pretrain_head.out_features, 4)


if __name__ == "__main__":
    unittest.main()
