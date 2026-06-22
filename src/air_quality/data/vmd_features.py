"""VMD-AQI 特征变换工具

提供两种 VMD 特征变换接口：
- ``apply_vmd_to_aqi``  (兼容旧实现)：只对 AQI 列做 VMD，输入 (N, T, 8+K)
- ``apply_vmd_to_features`` (推荐)：对指定污染物做 VMD（默认全部 7 个），
  解决「VMD 只分解 AQI 时其他污染物（特别是 O3）缺少频率分解信息、
  模型被迫均值回归」的根本问题。

数据流：
    原始 (N, T, F) → apply_vmd_to_features → (N, T, F + K - 1) 或 (N, T, 7K + 2)

注意：VMD 应当作用于 **原始尺度** 信号，而不是标准化后的信号——
标准化会引入负值，破坏 VMD 对物理信号的窄带假设。调用方需传入 raw scale 输入。
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .vmd import VMDDecomposer

# 7 个污染物的标准顺序（与 AirQualityDataProcessor.feature_columns 对齐）
POLLUTANT_ORDER: List[str] = [
    'aqi', 'pm2_5_24h', 'pm10_24h', 'no2_24h', 'so2_24h', 'co_24h', 'o3_8h_24h',
]


def apply_vmd_to_aqi(X: np.ndarray, decomposer: VMDDecomposer) -> np.ndarray:
    """对每个时间序列的 AQI 列（最后一维的第 0 列）做 VMD 分解。

    旧接口（兼容保留）：仅分解 AQI 列，其他污染物保留原值。

    Args:
        X: (N, T, 9) — 第 0 维是 AQI（**原始尺度**为佳，scaled 也可）
        decomposer: VMDDecomposer 实例

    Returns:
        (N, T, 8 + K) — 第 0..K-1 维是 K 个 IMF 分量，第 K..K+7 维是其余 8 维原始特征
    """
    N, T, _ = X.shape
    aqi_signals = X[:, :, 0]  # (N, T)
    imfs = decomposer.decompose(aqi_signals)  # (N, K, T)
    imfs = imfs.transpose(0, 2, 1)  # (N, T, K)
    other_features = X[:, :, 1:]  # (N, T, 8)
    return np.concatenate([imfs, other_features], axis=-1)


def apply_vmd_to_features(
    X: np.ndarray,
    decomposer: VMDDecomposer,
    target_features: Optional[Sequence[str]] = None,
    pollutant_order: Sequence[str] = POLLUTANT_ORDER,
    keep_raw: bool = False,
) -> np.ndarray:
    """对指定污染物做 VMD 分解，生成频率分量特征。

    推荐用法：用此函数替代 ``apply_vmd_to_aqi``，让模型能从每个污染物的
    独立频率分解中学习，避免因「AQI 与 O3 反相关」而被迫均值回归。

    输出按顺序拼接：所有目标污染物的 K 个 IMF + calendar 特征（month/season）。
    默认 ``target_features=None`` 等价于 ``pollutant_order``，即对全部 7 个
    污染物做 VMD。

    Args:
        X: (N, T, F) — 输入序列，至少包含目标污染物所在列 + month/season
        decomposer: VMDDecomposer 实例
        target_features: 要做 VMD 的污染物名列表；None 表示全部 7 个
        pollutant_order: X 中前 7 列对应的污染物名顺序，必须与 ``X`` 的列布局匹配
        keep_raw: 是否在 IMF 之外保留原始特征列（默认 False，仅保留日历）

    Returns:
        (N, T, K * len(target_features) + (2 if keep_raw else 0))

        IMF 顺序：依次遍历 ``target_features``，每个污染物先排放其 K 个 IMF。
        若 ``keep_raw=True``，则在 IMF 之后追加原始特征的前 ``len(target_features)``
        列（即目标污染物的 raw 值），再追加 calendar。

    维度示例（target_features=None, K=4）：
        输入 (N, T, 9) → 输出 (N, T, 7*4) = (N, T, 28)
        加上 calendar：实际使用时再额外拼接 (N, T, 2)，共 30 维。
    """
    if target_features is None:
        target_features = list(pollutant_order)
    else:
        # 校验 target_features 是 pollutant_order 的子集
        unknown = [f for f in target_features if f not in pollutant_order]
        if unknown:
            raise ValueError(
                f"target_features 含未知污染物: {unknown}; 必须是 {list(pollutant_order)} 的子集"
            )

    K = decomposer.K
    poll_idx = {name: i for i, name in enumerate(pollutant_order)}
    imfs_per_feature: List[np.ndarray] = []
    for feat in target_features:
        col = X[:, :, poll_idx[feat]]  # (N, T)
        imfs = decomposer.decompose(col)  # (N, K, T)
        imfs_per_feature.append(imfs.transpose(0, 2, 1))  # (N, T, K)

    imfs_all = np.concatenate(imfs_per_feature, axis=-1)  # (N, T, K*len(target))

    parts: List[np.ndarray] = [imfs_all]
    if keep_raw:
        # 仅保留目标污染物的原始列
        raw_target = np.stack([X[:, :, poll_idx[f]] for f in target_features], axis=-1)
        parts.append(raw_target)
    # calendar 始终保留（无论 keep_raw 与否）
    # 现在 calendar 是 4 个周期性编码特征（month_sin, month_cos, season_sin, season_cos）
    parts.append(X[:, :, -4:])  # (N, T, 4)
    return np.concatenate(parts, axis=-1)