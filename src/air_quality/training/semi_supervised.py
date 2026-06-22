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


def compute_pseudo_confidence(
    predictions: np.ndarray,
    target_std: np.ndarray = None,
) -> np.ndarray:
    """计算每样本的伪标签置信度。

    设计动机：旧公式 ``1/(1+std)`` 会让 Teacher 的「过度平滑预测」获得高置信度，
    形成「平滑预测 → 高置信 → 选为伪标签 → Student 学到同样平滑」的正反馈陷阱，
    是半监督 VMD 模型均值回归的关键诱因。

    新公式：基于预测 std 与目标 std 的接近程度计算置信度——
    - 当 ``std(pred) ≈ std(target)`` 时，置信度最高（≈1）
    - 当 ``std(pred) ≪ std(target)``（过度平滑）或 ``std(pred) ≫ std(target)``（噪声）
      时，置信度都衰减

    Args:
        predictions: (N, T, output_size)
        target_std: (output_size,) 各特征目标 std；若为 None 则用全局统计的中位数估计

    Returns:
        (N,) 置信度分数，范围 (0, 1]
    """
    std_per_sample = predictions.std(axis=(1, 2))  # (N,) — 每样本所有时间步+特征的总波动

    if target_std is None:
        # 退化路径：用 predictions 自身的中位数 std 作为锚点，
        # 至少可以抑制「远低于多数样本」的过度平滑预测
        anchor = float(np.median(std_per_sample))
    else:
        # target_std 是 (F,)，先合并为单一标量再做比较
        anchor = float(np.mean(target_std))

    eps = 1e-8
    # 比值 = pred_std / anchor；=1 表示波动幅度匹配目标
    ratio = std_per_sample / (anchor + eps)
    # 高斯核形式：ratio=1 时得分最大；偏离 1 越多得分越低
    confidence = np.exp(-((ratio - 1.0) ** 2))
    return confidence


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
        weight_decay: L2 正则化系数

    损失 / 反平滑（与全监督 ``ModelTrainer`` 对齐，避免半监督模型均值回归）:
        loss_type: 损失函数类型，默认 'mse_antismooth'（与全监督一致）
        loss_kwargs: 反平滑损失参数（lambda_var/lambda_diff/tau_*/lambda_warmup_epochs）
        detect_smoothing: 是否启用反平滑检测作为额外早停信号
        smoothing_threshold / smoothing_stop_patience: 反平滑检测阈值与早停耐心值
        gradient_clip: 梯度裁剪阈值
        early_stop_patience: Teacher/Student 共用的 test-loss 早停耐心值
    """

    # 与 ModelTrainer 保持一致的默认值。train.py 会从 config 透传真实覆盖值。
    _DEFAULT_LOSS_TYPE = 'mse_antismooth'
    _DEFAULT_LOSS_KWARGS: Dict[str, Any] = {
        'delta': 1.0,
        'lambda_var': 0.1,
        'lambda_diff': 0.05,
        'tau_var': 0.5,
        'tau_diff': 0.5,
        'lambda_warmup_epochs': 20,
    }

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
        loss_type: str = None,
        loss_kwargs: Optional[Dict[str, Any]] = None,
        detect_smoothing: bool = True,
        smoothing_threshold: float = 0.1,
        smoothing_stop_patience: int = 15,
        gradient_clip: float = 1.0,
        early_stop_patience: int = 30,
        vmd_K: int = None,
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
        # 损失 / 反平滑：与全监督 ModelTrainer 使用同一套配置，
        # 这是修复"半监督预测过度平滑"的关键——此前硬编码 huber 完全绕过了反平滑损失。
        self.loss_type = loss_type or self._DEFAULT_LOSS_TYPE
        self.loss_kwargs = dict(self._DEFAULT_LOSS_KWARGS)
        if loss_kwargs:
            self.loss_kwargs.update(loss_kwargs)
        self.detect_smoothing = detect_smoothing
        self.smoothing_threshold = smoothing_threshold
        self.smoothing_stop_patience = smoothing_stop_patience
        self.gradient_clip = gradient_clip
        self.early_stop_patience = early_stop_patience
        self.vmd_K = vmd_K  # VMD 模态数，传给 create_model 保持与数据一致
        self.history: Dict[str, Any] = {
            'teacher_losses': [],
            'student_losses': [],
            'pseudo_label_rate': [],
            # 与全监督对齐的聚合字段（供 plot_training_history 渲染 loss/lr/精度面板）
            'teacher_test_loss': [],
            'teacher_lr': [],
            'student_test_loss': [],
            'student_lr': [],
            'smoothing_flags_per_epoch': [],  # 跨阶段顺序拼接 (teacher→student)
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
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        """创建模型,若 backbone_state 不为空则加载到共享部分。

        model_kwargs 用于传入模型特定参数（如 vmd_K、bilstm_hidden 等），
        避免 vmd_K 默认值与数据实际 K 不匹配导致 pretrain_head 维度错位。
        """
        kwargs = dict(model_kwargs) if model_kwargs else {}
        model = create_model(model_type, input_size=input_size, **kwargs)
        if backbone_state is not None:
            # strict=False 允许 head 维度不匹配
            model.load_state_dict(backbone_state, strict=False)
        return model

    def _train_teacher(self, X_labeled, y_labeled, X_val, y_val):
        """Phase 1: Teacher 全监督预训练"""
        logger.info("[Phase 1] Training Teacher on labeled data...")
        backbone_state = getattr(self, '_backbone_state', None)
        # 将 vmd_K 传给 create_model，避免与数据 K 不匹配
        model_kwargs = {'vmd_K': self.vmd_K} if self.vmd_K is not None else None
        teacher = self._build_model_with_state(
            self.model_type, input_size=self.input_size,
            backbone_state=backbone_state,
            model_kwargs=model_kwargs,
        ).to(self.device)
        train_loader = self._make_loader(X_labeled, y_labeled, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False) if X_val is not None else None

        trainer = ModelTrainer(
            model=teacher,
            train_loader=train_loader,
            test_loader=val_loader,
            device=self.device,
            best_model_path=None,
            loss_type=self.loss_type,
            early_stop_patience=self.early_stop_patience,
            gradient_clip=self.gradient_clip,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            loss_kwargs=self.loss_kwargs,
            detect_smoothing=self.detect_smoothing,
            smoothing_threshold=self.smoothing_threshold,
            smoothing_stop_patience=self.smoothing_stop_patience,
        )
        trainer.train(num_epochs=self.teacher_epochs)
        self._record_phase_history(trainer.history, phase='teacher')
        # 传播训练摘要（供 log_training_summary 使用）
        self._last_trainer_history = trainer.history
        return teacher

    def _record_phase_history(self, trainer_history: Dict[str, Any], phase: str) -> None:
        """把单阶段 ModelTrainer.history 的 train/test loss、lr、反平滑标记
        聚合进 self.history，使半监督图表与全监督拥有同等的字段（修复
        "字段不匹配导致图表不一致"）。

        - ``<phase>_losses`` / ``<phase>_test_loss`` / ``<phase>_lr``：本阶段曲线
        - ``smoothing_flags_per_epoch``：跨阶段顺序拼接，记录每 epoch 被标记的
          污染物数，用于在图表上展示反平滑检测信号
        """
        losses = trainer_history.get('train_loss', [])
        self.history[f'{phase}_losses'].extend(losses)
        self.history[f'{phase}_test_loss'].extend(trainer_history.get('test_loss', []))
        self.history[f'{phase}_lr'].extend(trainer_history.get('lr', []))
        # smoothing_flags_per_epoch 是每阶段 list[dict]，按阶段顺序拼成一条时间轴
        flags = trainer_history.get('smoothing_flags_per_epoch', [])
        if flags:
            self.history['smoothing_flags_per_epoch'].extend(flags)

    def _generate_pseudo_labels(self, teacher, X_unlabeled, X_labeled=None):
        """Phase 2: Teacher 生成伪标签

        置信度公式基于「预测 std 与目标 std 的匹配度」，
        避免旧版「std 越低 → 置信度越高」的正反馈陷阱。
        若提供了 X_labeled，则用其每个时间窗口的 std 作为目标锚点；
        否则用 predictions 自身的中位数作为锚点（退化路径）。
        """
        logger.info("[Phase 2] Generating pseudo labels...")
        teacher.eval()
        with torch.no_grad():
            x = torch.FloatTensor(X_unlabeled).to(self.device)
            preds = teacher(x).cpu().numpy()  # (N, T, output_size)

        # 用 labeled 数据的真实 std 作为置信度锚点
        target_std = None
        if X_labeled is not None and len(X_labeled) > 0:
            # X_labeled 是已经经过 VMD 处理的特征；其最后 7 维对应 7 个污染物目标
            target_std = X_labeled[..., :7].std(axis=(0, 1))  # (7,)

        confidences = compute_pseudo_confidence(preds, target_std=target_std)
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
        model_kwargs = {'vmd_K': self.vmd_K} if self.vmd_K is not None else None
        student = self._build_model_with_state(
            self.model_type, input_size=self.input_size,
            backbone_state=backbone_state,
            model_kwargs=model_kwargs,
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
            loss_type=self.loss_type,
            early_stop_patience=self.early_stop_patience,
            gradient_clip=self.gradient_clip,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            loss_kwargs=self.loss_kwargs,
            detect_smoothing=self.detect_smoothing,
            smoothing_threshold=self.smoothing_threshold,
            smoothing_stop_patience=self.smoothing_stop_patience,
        )
        trainer.train(num_epochs=self.student_epochs)
        self._record_phase_history(trainer.history, phase='student')
        # 传播训练摘要（供 log_training_summary 使用）
        self._last_trainer_history = trainer.history
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
        pseudo_preds, mask = self._generate_pseudo_labels(teacher, X_unlabeled, X_labeled=X_labeled)
        student = self._train_student(X_labeled, y_labeled, X_unlabeled, pseudo_preds, mask)
        metrics = self._evaluate(student, X_test, y_test)

        # 将内部训练的摘要信息写入 history（供 log_training_summary 使用）
        if hasattr(self, '_last_trainer_history'):
            th = self._last_trainer_history
            self.history['epochs_trained'] = th.get('epochs_trained', 0)
            self.history['best_epoch'] = th.get('best_epoch', 0)
            self.history['best_loss'] = th.get('best_loss', float('inf'))
            self.history['smoothing_stopped'] = th.get('smoothing_stopped', False)

        return student, metrics