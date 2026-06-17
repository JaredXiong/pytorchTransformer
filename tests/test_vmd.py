"""VMD 分解器单元测试"""
import unittest
import numpy as np

from air_quality.data.vmd import VMDDecomposer


class TestVMDDecomposer(unittest.TestCase):
    def test_decompose_returns_K_modes(self):
        """应返回 K 个 IMF 分量"""
        decomposer = VMDDecomposer(K=4)
        signal = np.sin(np.linspace(0, 4 * np.pi, 100))
        result = decomposer.decompose(signal)
        self.assertEqual(result.shape, (4, 100))

    def test_decompose_reconstruction_error(self):
        """重构误差应在可接受范围内（< 15%，符合 VMD 文献常见阈值）"""
        decomposer = VMDDecomposer(K=4, alpha=2000)
        np.random.seed(42)
        signal = np.sin(np.linspace(0, 4 * np.pi, 100)) + 0.1 * np.random.randn(100)
        imfs = decomposer.decompose(signal)
        reconstructed = imfs.sum(axis=0)
        error = np.linalg.norm(signal - reconstructed) / np.linalg.norm(signal)
        self.assertLess(error, 0.15)  # 重构误差 < 15% (VMD 常见可接受范围)

    def test_decompose_2d_input(self):
        """二维输入 (N, T) 应逐行分解"""
        decomposer = VMDDecomposer(K=4)
        signals = np.random.randn(3, 100)
        result = decomposer.decompose(signals)
        self.assertEqual(result.shape, (3, 4, 100))

    def test_configurable_K(self):
        """K 参数应可配置"""
        for K in [2, 4, 6]:
            decomposer = VMDDecomposer(K=K)
            signal = np.sin(np.linspace(0, 4 * np.pi, 100))
            result = decomposer.decompose(signal)
            self.assertEqual(result.shape, (K, 100))
