"""
Transformer 模型
"""
import torch
import torch.nn as nn
import math
from .base import BaseModel, PositionalEncoding


class TransformerModel(BaseModel):
    """Transformer模型，用于时序数据预测"""

    def __init__(self, input_size: int, output_size: int = 7, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 4, dropout: float = 0.2):
        super().__init__(input_size, output_size, dropout)
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        self.encoder = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=512,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, output_size)
        self.init_weights()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = self.dropout(src)
        output = self.transformer_encoder(src)
        return self.decoder(output)
