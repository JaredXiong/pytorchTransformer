"""
PretrainFinetuneTrainer:三阶段流水线编排器

继承 SemiSupervisedTrainer,在 fit() 之前增加 Phase 0(无监督预训练)。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from air_quality.config import config as global_config
from air_quality.config.settings import PretrainConfig
from air_quality.models import create_model
from air_quality.training.pretrain import Pretrainer
from air_quality.training.semi_supervised import SemiSupervisedTrainer

logger = logging.getLogger(__name__)


class PretrainFinetuneTrainer(SemiSupervisedTrainer):
    """三阶段流水线(预训练 → 伪标签 → 微调)"""

    def __init__(
        self,
        model_type: str,
        input_size: int,
        device: str = 'cpu',
        teacher_epochs: int = 80,
        student_epochs: int = 120,
        pseudo_confidence_threshold: float = 0.85,
        batch_size: int = 32,
        learning_rate: float = 3e-4,
        weight_decay: float = 3e-4,
        pretrain_config: Optional[PretrainConfig] = None,
    ):
        super().__init__(
            model_type=model_type,
            input_size=input_size,
            device=device,
            teacher_epochs=teacher_epochs,
            student_epochs=student_epochs,
            pseudo_confidence_threshold=pseudo_confidence_threshold,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        # 默认从全局 config 读取,允许显式覆盖
        self.pretrain_config: PretrainConfig = pretrain_config or global_config.pretrain
        # 新增 history 字段
        self.history.setdefault('pretrain_losses', [])
        # 显式声明 _backbone_state(避免 getattr hack)
        self._backbone_state: Optional[Dict[str, torch.Tensor]] = None
        # 暂存预训练模型(供测试访问)
        self._pretrain_model: Optional[nn.Module] = None

    def _run_pretrain(self, X_unlabeled: np.ndarray) -> Optional[Dict[str, torch.Tensor]]:
        """Phase 0:预训练;返回 backbone_state 或 None。"""
        if not self.pretrain_config.enabled:
            return None

        logger.info("[Phase 0] Pretraining on unlabeled data...")
        model = create_model(self.model_type, input_size=self.input_size)
        pretrainer = Pretrainer(
            model=model,
            device=str(self.device),
            learning_rate=self.pretrain_config.learning_rate,
            weight_decay=self.pretrain_config.weight_decay,
            mask_ratio=self.pretrain_config.mask_ratio,
            early_stop_patience=self.pretrain_config.early_stop_patience,
        )
        pretrainer.fit(X_unlabeled, epochs=self.pretrain_config.epochs, batch_size=self.pretrain_config.batch_size)
        self.history['pretrain_losses'] = pretrainer.history['train_losses']
        self._pretrain_model = model
        return Pretrainer.extract_backbone_state(model)

    def fit(
        self,
        X_labeled,
        y_labeled,
        X_unlabeled,
        y_unlabeled,
        X_test,
        y_test,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        # Phase 0: 预训练
        self._backbone_state = self._run_pretrain(X_unlabeled)

        # Phase 1-4: 沿用 SemiSupervisedTrainer(通过 _fit_semi 钩子便于子类/测试拦截)
        return self._fit_semi(
            X_labeled, y_labeled,
            X_unlabeled, y_unlabeled,
            X_test, y_test,
        )

    # 暴露给测试使用(原 plan 称之为 _fit_semi;实际是 super().fit 的别称)
    def _fit_semi(self, *args, **kwargs):
        """内部钩子:委托父类 fit 4 阶段流水线。

        存在此方法是为了让测试可以 monkey-patch 它,验证
        pretrain backbone 是否被正确加载到 Teacher/Student。
        生产代码应直接调用 self.fit(...),无需触碰此方法。
        """
        return super().fit(*args, **kwargs)
