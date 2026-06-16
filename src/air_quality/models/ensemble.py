"""
集成模型 - 融合多个模型的预测
"""
import torch
import torch.nn as nn
from typing import List, Optional


class EnsembleModel(nn.Module):
    """集成模型 - 融合多个模型的预测"""

    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)

        # 加权平均
        stacked = torch.stack(predictions)
        weights = self.weights.view(-1, 1, 1, 1)
        return (stacked * weights).sum(dim=0)