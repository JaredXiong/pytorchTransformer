"""VMD-AQI 特征变换工具

将原始 (N, T, 9) 特征中的 AQI 列（第 0 维）替换为 K 个 VMD IMF 分量，
返回 (N, T, 8+K) 特征。
"""
import numpy as np

from .vmd import VMDDecomposer


def apply_vmd_to_aqi(X: np.ndarray, decomposer: VMDDecomposer) -> np.ndarray:
    """对每个时间序列的 AQI 列（最后一维的第 0 列）做 VMD 分解。

    Args:
        X: (N, T, 9) — 第 0 维是 AQI
        decomposer: VMDDecomposer 实例

    Returns:
        (N, T, 8 + K) — 第 0..K-1 维是 K 个 IMF 分量，第 K..K+7 维是其余 8 维原始特征
    """
    N, T, _ = X.shape
    aqi_signals = X[:, :, 0]  # (N, T)
    imfs = decomposer.decompose(aqi_signals)  # (N, K, T)
    # 转置为 (N, T, K) 以便拼接
    imfs = imfs.transpose(0, 2, 1)
    other_features = X[:, :, 1:]  # (N, T, 8)
    return np.concatenate([imfs, other_features], axis=-1)
