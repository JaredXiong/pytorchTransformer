"""
LSTM 与 GRU 模型
"""
import torch
import torch.nn as nn
from .base import BaseModel


class LSTMModel(BaseModel):
    """LSTM模型，用于时序数据预测"""

    def __init__(self, input_size: int, output_size: int = 7, hidden_size: int = 256,
                 num_layers: int = 4, dropout: float = 0.05, bidirectional: bool = True):
        super().__init__(input_size, output_size, dropout)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.fc1 = nn.Linear(hidden_size * self.num_directions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        out = self.relu(self.fc1(lstm_out))
        out = self.dropout(out)
        return self.fc2(out)


