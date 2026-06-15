"""
模型训练器
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ModelTrainer:
    """模型训练器"""

    def __init__(self, model, train_loader, test_loader, device,
                 best_model_path: str = None, loss_type: str = 'huber',
                 early_stop_patience: int = 15, gradient_clip: float = 1.0,
                 learning_rate: float = 0.001):
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
            weight_decay=1e-5
        )

        # 根据类型获取损失函数
        if loss_type == 'huber':
            self.criterion = nn.HuberLoss(delta=1.0)
        elif loss_type == 'mae':
            self.criterion = nn.L1Loss()
        elif loss_type == 'mse':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.HuberLoss(delta=1.0)

        self.scheduler = None
        self.best_loss = float('inf')
        self.history = {
            'train_loss': [],
            'test_loss': [],
            'best_epoch': 0,
            'best_loss': float('inf'),
            'lr': [],
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

        for epoch in range(num_epochs):
            train_loss = self._train_epoch()
            test_loss = self._validate_epoch()

            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)
            self.history['lr'].append(current_lr)

            if self.scheduler:
                self.scheduler.step()

            if test_loss < self.best_loss:
                self.best_loss = test_loss
                self.history['best_epoch'] = epoch
                self.history['best_loss'] = test_loss
                early_stop_counter = 0
                self._save_model()
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    logger.info(f"提前停止训练，第{epoch + 1}轮")
                    break

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, "
                    f"LR: {current_lr:.6f}"
                )

        self.history['epochs_trained'] = epoch + 1

    def _train_epoch(self) -> float:
        """单轮训练"""
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

        return total_loss / batch_count

    def _validate_epoch(self) -> float:
        """单轮验证"""
        self.model.eval()
        total_loss = 0.0
        batch_count = 0

        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                output = self.model(batch_x)
                prediction_days = batch_y.size(1)
                output = output[:, -prediction_days:, :]
                loss = self.criterion(output, batch_y)
                total_loss += loss.item()
                batch_count += 1

        return total_loss / batch_count

    def _save_model(self) -> None:
        """保存模型"""
        if self.best_model_path:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_loss': self.best_loss,
                'model_type': self.model.__class__.__name__,
            }, self.best_model_path)
