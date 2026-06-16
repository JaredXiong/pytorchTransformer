"""
模型训练器
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import logging
from typing import Optional, List, Dict, Any

from .losses import get_loss_function, AntiSmoothingLoss

logger = logging.getLogger(__name__)


# 7 个污染物的标准顺序，detect_over_smoothing 使用
DEFAULT_FEATURE_NAMES = ['aqi', 'pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3']


class ModelTrainer:
    """模型训练器"""

    def __init__(self, model, train_loader, test_loader, device,
                 best_model_path: str = None, loss_type: str = 'huber',
                 early_stop_patience: int = 15, gradient_clip: float = 1.0,
                 learning_rate: float = 0.001, weight_decay: float = 1e-5,
                 loss_kwargs: Optional[Dict[str, Any]] = None,
                 detect_smoothing: bool = False, smoothing_threshold: float = 0.1,
                 smoothing_stop_patience: int = 15, feature_names: Optional[List[str]] = None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.best_model_path = best_model_path
        self.early_stop_patience = early_stop_patience
        self.gradient_clip = gradient_clip
        self.learning_rate = learning_rate

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 通过损失工厂获取损失函数（统一入口）
        # 旧调用：硬编码 switch；新调用：get_loss_function(loss_type, **loss_kwargs)
        # 兼容旧行为：loss_kwargs=None 时仅传 delta 兜底
        if loss_kwargs is None:
            loss_kwargs = {'delta': 1.0}
        self.criterion = get_loss_function(loss_type, **loss_kwargs)
        # AntiSmoothingLoss 支持 epoch 进度同步（用于 warmup）
        if isinstance(self.criterion, AntiSmoothingLoss):
            self._is_anti_smooth = True
        else:
            self._is_anti_smooth = False

        # 反平滑检测
        self.detect_smoothing = detect_smoothing
        self.smoothing_threshold = smoothing_threshold
        self.smoothing_stop_patience = smoothing_stop_patience
        self._metrics_tracker = None
        if self.detect_smoothing:
            try:
                from .metrics import MetricsTracker
                self._metrics_tracker = MetricsTracker(
                    feature_names=feature_names or DEFAULT_FEATURE_NAMES
                )
            except Exception as e:  # pragma: no cover - 防御性
                logger.warning("无法初始化 MetricsTracker，detect_smoothing 关闭: %s", e)
                self.detect_smoothing = False
                self._metrics_tracker = None

        self.scheduler = None
        self.best_loss = float('inf')
        self._smoothing_counter = 0
        self.history = {
            'train_loss': [],
            'test_loss': [],
            'best_epoch': 0,
            'best_loss': float('inf'),
            'lr': [],
            'smoothing_flags_per_epoch': [],
        }

    def train(self, num_epochs: int, patience: int = None, use_scheduler: bool = True) -> None:
        """训练模型"""
        patience = patience or self.early_stop_patience

        # 设置学习率调度器
        if use_scheduler:
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate * 10,
                epochs=num_epochs,
                steps_per_epoch=len(self.train_loader),
                pct_start=0.1,
                anneal_strategy='cos'
            )

        early_stop_counter = 0
        smoothing_stopped = False

        for epoch in range(num_epochs):
            # 同步 warmup 进度到 AntiSmoothingLoss
            if self._is_anti_smooth:
                self.criterion.set_epoch(epoch)

            train_loss, _ = self._train_epoch()
            test_loss, val_payload = self._validate_epoch()

            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)
            self.history['lr'].append(current_lr)

            # 反平滑检测：统计被标记的污染物数
            over_smooth = val_payload.get('over_smooth', {}) if val_payload else {}
            n_flagged = sum(1 for v in over_smooth.values() if v)
            self.history['smoothing_flags_per_epoch'].append({
                'n_flagged': n_flagged,
                'flags': dict(over_smooth),
            })

            # 注意：scheduler.step() 已在 _train_epoch 每个 batch 末尾调用，
            # 不再在 epoch 末尾额外调用，避免 OneCycleLR 步数超限。

            if test_loss < self.best_loss:
                self.best_loss = test_loss
                self.history['best_epoch'] = epoch
                self.history['best_loss'] = test_loss
                early_stop_counter = 0
                self._save_model()
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    logger.info(f"提前停止训练，第{epoch + 1}轮（test loss 早停）")
                    break

            # 反平滑额外早停：连续 smoothing_stop_patience 个 epoch 有 ≥5/7 标记
            if self.detect_smoothing and len(over_smooth) > 0:
                total_features = len(over_smooth)
                # 当被标记数超过半数时累加 counter，反之重置
                if n_flagged >= max(1, total_features // 2 + 1):
                    self._smoothing_counter += 1
                else:
                    self._smoothing_counter = max(0, self._smoothing_counter - 1)
                if self._smoothing_counter >= self.smoothing_stop_patience:
                    logger.info(
                        f"提前停止训练，第{epoch + 1}轮（连续 {self._smoothing_counter} 个 epoch "
                        f"有 {n_flagged}/{total_features} 个污染物被标记为过度平滑）"
                    )
                    smoothing_stopped = True
                    break

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, "
                    f"LR: {current_lr:.6f}, Over-smooth: {n_flagged}/{len(over_smooth)}"
                )

        self.history['epochs_trained'] = epoch + 1
        self.history['smoothing_stopped'] = smoothing_stopped

    def _train_epoch(self):
        """单轮训练，返回 (avg_loss, None)"""
        self.model.train()
        total_loss = 0.0
        batch_count = 0

        for batch_x, batch_y in self.train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(batch_x)
            prediction_days = batch_y.size(1)
            output = output[:, -prediction_days:, :]
            loss = self.criterion(output, batch_y)
            loss.backward()

            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()
            batch_count += 1

        return total_loss / batch_count, None

    def _validate_epoch(self):
        """单轮验证，返回 (avg_loss, payload)。
        payload 包含：
        - 'y_true': np.ndarray (N_val, T_pred, F)
        - 'y_pred': np.ndarray (N_val, T_pred, F)
        - 'over_smooth': dict[feature_name, bool]（仅当 detect_smoothing=True 时）
        """
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        y_true_chunks: List = []
        y_pred_chunks: List = []

        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                output = self.model(batch_x)
                prediction_days = batch_y.size(1)
                sliced_out = output[:, -prediction_days:, :]
                loss = self.criterion(sliced_out, batch_y)
                total_loss += loss.item()
                batch_count += 1
                if self.detect_smoothing:
                    # CPU + numpy；保留 (N, T, F) 以便 std/diff 沿时间轴
                    y_pred_chunks.append(sliced_out.detach().cpu().numpy())
                    y_true_chunks.append(batch_y.detach().cpu().numpy())

        payload: Dict[str, Any] = {}
        if self.detect_smoothing and y_pred_chunks:
            import numpy as np
            y_true_arr = np.concatenate(y_true_chunks, axis=0)
            y_pred_arr = np.concatenate(y_pred_chunks, axis=0)
            payload['y_true'] = y_true_arr
            payload['y_pred'] = y_pred_arr
            try:
                over_smooth = self._metrics_tracker.detect_over_smoothing(
                    y_true_arr, y_pred_arr, threshold=self.smoothing_threshold
                )
                payload['over_smooth'] = over_smooth
            except Exception as e:  # pragma: no cover
                logger.warning("detect_over_smoothing 失败: %s", e)
                payload['over_smooth'] = {}

        return total_loss / batch_count, payload

    def _save_model(self) -> None:
        """保存模型"""
        if self.best_model_path:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_loss': self.best_loss,
                'model_type': self.model.__class__.__name__,
            }, self.best_model_path)
