"""VMD (Variational Mode Decomposition) 分解器

封装 vmdpy 库，提供对一维与二维信号的变分模态分解。
"""
from __future__ import annotations

import numpy as np
from vmdpy import VMD


class VMDDecomposer:
    """VMD 分解器

    Args:
        K: IMF 模态数（默认 4）
        alpha: 带宽约束（默认 2000，值越大带宽越窄）
        tau: 噪声容忍（默认 0）
        DC: 是否保留 DC 分量（默认 0 = 不保留）
        init: 初始化方式（1=均匀分布，2=随机）
        tol: 收敛容差（默认 1e-7）
    """

    def __init__(
        self,
        K: int = 4,
        alpha: float = 2000,
        tau: float = 0,
        DC: int = 0,
        init: int = 1,
        tol: float = 1e-7,
    ):
        if K < 1:
            raise ValueError(f"K 必须 ≥ 1，当前值: {K}")
        self.K = K
        self.alpha = alpha
        self.tau = tau
        self.DC = DC
        self.init = init
        self.tol = tol

    def decompose(self, signal: np.ndarray) -> np.ndarray:
        """对信号做 VMD 分解

        Args:
            signal: 一维 (T,) 或二维 (N, T) 信号

        Returns:
            一维输入返回 (K, T)；二维输入返回 (N, K, T)
        """
        if signal.ndim == 1:
            return self._decompose_single(signal)
        elif signal.ndim == 2:
            return np.stack(
                [self._decompose_single(row) for row in signal], axis=0
            )
        else:
            raise ValueError(f"signal 维度必须是 1 或 2，当前: {signal.ndim}")

    def _decompose_single(self, signal: np.ndarray) -> np.ndarray:
        u, _, _ = VMD(
            signal,
            self.alpha,
            self.tau,
            self.K,
            self.DC,
            self.init,
            self.tol,
        )
        return u  # (K, T)
