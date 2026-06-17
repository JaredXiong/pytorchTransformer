"""
SemiSupervisedTrainer

实现伪标签半监督学习的 4 阶段训练流程：
  Phase 1: Teacher 在 labeled 数据上全监督预训练
  Phase 2: Teacher 对 unlabeled 数据生成伪标签（高置信度过滤）
  Phase 3: Student 在 labeled + 伪标签 unlabeled 上联合训练
  Phase 4: 在 test 数据上评估
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tud

from air_quality.models import create_model
from air_quality.training.trainer import ModelTrainer

logger = logging.getLogger(__name__)


def compute_pseudo_confidence(predictions: np.ndarray) -> np.ndarray:
    """计算每样本的伪标签置信度。

    Args:
        predictions: (N, T, output_size)

    Returns:
        (N,) 置信度分数，范围 [0, 1]。低方差 → 高置信度。
    """
    std_per_sample = predictions.std(axis=(1, 2))  # (N,)
    return 1.0 / (1.0 + std_per_sample)


class SemiSupervisedTrainer:
    """伪标签半监督训练器

    Args:
        model_type: 模型类型字符串（如 'vmd_cnn_bilstm_attention'）
        input_size: 输入特征维度
        device: 'cuda' / 'cpu'
        teacher_epochs: Teacher 训练轮数
        student_epochs: Student 训练轮数
        pseudo_confidence_threshold: 伪标签置信度阈值
        batch_size: DataLoader batch size
        learning_rate: 学习率
    """

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
    ):
        self.model_type = model_type
        self.input_size = input_size
        self.device = torch.device(device)
        self.teacher_epochs = teacher_epochs
        self.student_epochs = student_epochs
        self.pseudo_threshold = pseudo_confidence_threshold
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.history: Dict[str, Any] = {
            'teacher_losses': [],
            'student_losses': [],
            'pseudo_label_rate': [],
        }

    def _make_loader(self, X, y, shuffle=True):
        return tud.DataLoader(
            tud.TensorDataset(
                torch.FloatTensor(X),
                torch.FloatTensor(y),
            ),
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

    @staticmethod
    def _build_model_with_state(
        model_type: str,
        input_size: int,
        backbone_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> nn.Module:
        """创建模型,若 backbone_state 不为空则加载到共享部分。"""
        model = create_model(model_type, input_size=input_size)
        if backbone_state is not None:
            # strict=False 允许 head 维度不匹配
            model.load_state_dict(backbone_state, strict=False)
        return model

    def _train_teacher(self, X_labeled, y_labeled, X_val, y_val):
        """Phase 1: Teacher 全监督预训练"""
        logger.info("[Phase 1] Training Teacher on labeled data...")
        backbone_state = getattr(self, '_backbone_state', None)
        teacher = self._build_model_with_state(
            self.model_type, input_size=self.input_size,
            backbone_state=backbone_state,
        ).to(self.device)
        train_loader = self._make_loader(X_labeled, y_labeled, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False) if X_val is not None else None

        trainer = ModelTrainer(
            model=teacher,
            train_loader=train_loader,
            test_loader=val_loader,
            device=self.device,
            best_model_path=None,
            loss_type='huber',
            early_stop_patience=self.teacher_epochs,
            gradient_clip=1.0,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        trainer.train(num_epochs=self.teacher_epochs)
        self.history['teacher_losses'].extend(trainer.history.get('train_loss', []))
        return teacher

    def _generate_pseudo_labels(self, teacher, X_unlabeled):
        """Phase 2: Teacher 生成伪标签"""
        logger.info("[Phase 2] Generating pseudo labels...")
        teacher.eval()
        with torch.no_grad():
            x = torch.FloatTensor(X_unlabeled).to(self.device)
            preds = teacher(x).cpu().numpy()  # (N, T, output_size)

        confidences = compute_pseudo_confidence(preds)
        mask = confidences >= self.pseudo_threshold
        rate = float(mask.mean())
        self.history['pseudo_label_rate'].append(rate)
        logger.info(f"  Pseudo label retention rate: {rate:.2%}")
        # 防止空集：若所有都被过滤，保留置信度最高的 10%
        if not mask.any():
            top_k = max(1, int(0.1 * len(mask)))
            top_indices = np.argsort(-confidences)[:top_k]
            mask = np.zeros_like(mask, dtype=bool)
            mask[top_indices] = True
        return preds, mask

    def _train_student(self, X_labeled, y_labeled, X_pseudo, y_pseudo, mask):
        """Phase 3: Student 联合训练"""
        logger.info("[Phase 3] Training Student on labeled + pseudo labels...")
        backbone_state = getattr(self, '_backbone_state', None)
        student = self._build_model_with_state(
            self.model_type, input_size=self.input_size,
            backbone_state=backbone_state,
        ).to(self.device)

        X_pseudo_filtered = X_pseudo[mask]
        # ModelTrainer 用 output[:, -prediction_days:, :] 与 target 对齐，
        # 因此 target 仅含预测窗口（最后 prediction_days 步）。
        prediction_days = y_labeled.shape[1]
        y_pseudo_filtered = y_pseudo[mask][:, -prediction_days:, :]
        X_combined = np.concatenate([X_labeled, X_pseudo_filtered], axis=0)
        y_combined = np.concatenate([y_labeled, y_pseudo_filtered], axis=0)

        train_loader = self._make_loader(X_combined, y_combined, shuffle=True)
        # 用训练集后 1 个样本作为占位验证集（ModelTrainer._validate_epoch 不接受 None）
        val_loader = self._make_loader(
            X_combined[-1:], y_combined[-1:], shuffle=False
        )
        trainer = ModelTrainer(
            model=student,
            train_loader=train_loader,
            test_loader=val_loader,
            device=self.device,
            best_model_path=None,
            loss_type='huber',
            early_stop_patience=self.student_epochs,
            gradient_clip=1.0,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        trainer.train(num_epochs=self.student_epochs)
        self.history['student_losses'].extend(trainer.history.get('train_loss', []))
        return student

    def _evaluate(self, model, X_test, y_test):
        """Phase 4: 测试集评估"""
        logger.info("[Phase 4] Evaluating on test set...")
        model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(X_test).to(self.device)
            preds = model(x).cpu().numpy()
        # 取最后 prediction_days 步（与 y_test 对齐）
        n_pred_steps = y_test.shape[1]
        preds_last = preds[:, -n_pred_steps:, :]
        loss = float(np.mean((preds_last - y_test) ** 2))
        return {'test_loss': loss, 'predictions': preds_last}

    def fit(
        self,
        X_labeled,
        y_labeled,
        X_unlabeled,
        y_unlabeled,
        X_test,
        y_test,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """完整 4 阶段训练流程"""
        # 使用 labeled 数据的最后 20% 作为 Teacher 验证集
        n_val = max(1, int(0.2 * len(X_labeled)))
        X_tr = X_labeled[:-n_val]
        y_tr = y_labeled[:-n_val]
        X_val = X_labeled[-n_val:]
        y_val = y_labeled[-n_val:]

        teacher = self._train_teacher(X_tr, y_tr, X_val, y_val)
        pseudo_preds, mask = self._generate_pseudo_labels(teacher, X_unlabeled)
        student = self._train_student(X_labeled, y_labeled, X_unlabeled, pseudo_preds, mask)
        metrics = self._evaluate(student, X_test, y_test)
        return student, metrics