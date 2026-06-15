"""
CNN 模型和 TCN (Temporal Convolutional Network)
"""
import torch
import torch.nn as nn
from typing import Optional
from .base import BaseModel


class CNNModel(BaseModel):
    """CNN模型，使用多尺度卷积核提取时序特征"""

    def __init__(self, input_size: int, output_size: int = 7, num_filters: int = 64,
                 kernel_sizes: Optional[list] = None, dropout: float = 0.2):
        super().__init__(input_size, output_size, dropout)
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes or [3, 5, 7]

        self.convs = nn.ModuleList()
        for kernel_size in self.kernel_sizes:
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=input_size, out_channels=num_filters,
                              kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(num_filters),
                    nn.ReLU(),
                    nn.MaxPool1d(2, stride=1, padding=1)
                )
            )

        self.total_filters = num_filters * len(self.kernel_sizes)
        self.fc1 = nn.Linear(self.total_filters, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        conv_outputs = [conv(x) for conv in self.convs]
        out = torch.cat(conv_outputs, dim=1)
        out = nn.AdaptiveAvgPool1d(x.size(2))(out)
        out = out.permute(0, 2, 1)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)


class TCNBlock(nn.Module):
    """TCN 残差块 - 使用膨胀因果卷积"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        # 残差连接
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(BaseModel):
    """TCN (Temporal Convolutional Network) 模型"""

    def __init__(self, input_size: int, num_filters: int = 64, num_layers: int = 4,
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__(input_size, dropout)
        self.num_filters = num_filters
        self.num_layers = num_layers

        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else num_filters
            layers.append(TCNBlock(in_ch, num_filters, kernel_size, dilation, dropout))

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_filters, input_size)
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        out = self.network(x)
        out = out.permute(0, 2, 1)
        return self.fc(out)
