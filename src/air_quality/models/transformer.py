"""
Transformer 模型

反平滑重构：
- input_proj / residual_proj / residual_weight(sigmoid 约束 ∈ (0,1)) / feature_scale
- pre-LN (norm_first=True) 稳定深层训练
- dim_feedforward = ffn_mult * d_model（默认 4，标准 transformer 比例）
- attn_dropout 默认 0（避免过度正则化）
"""
import torch
import torch.nn as nn
import math
from .base import BaseModel, PositionalEncoding


class TransformerModel(BaseModel):
    """Transformer模型，用于时序数据预测

    输出形状契约：forward(x) 返回 (B, T, output_size)，与输入等长
    """

    def __init__(self, input_size: int, output_size: int = 7, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 4, dropout: float = 0.1,
                 residual_weight: float = 0.3, feature_scale_init: float = 1.0,
                 ffn_mult: int = 4, norm_first: bool = True):
        super().__init__(input_size, output_size, dropout)
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.output_size = output_size

        # 输入投影（残差 identity 入口）
        self.input_proj = nn.Linear(input_size, d_model)
        # 把 d_model 投到 output 空间（残差 shortcut 需要同维度相加）
        self.residual_proj = nn.Linear(d_model, output_size)
        # 残差权重：直接以实数参数化，前向时用 sigmoid 约束 ∈ (0, 1)
        # 这样既能学到"几乎不用残差"也能学到"几乎完全用残差"
        # 默认 0.3 → sigmoid(0.3) ≈ 0.574（温和偏向使用残差）
        self.residual_weight = nn.Parameter(torch.tensor(residual_weight))
        # 逐特征门控：防止最终 Linear 把不同污染物压到同一均值
        self.feature_scale = nn.Parameter(torch.full((output_size,), feature_scale_init))

        self.pos_encoder = PositionalEncoding(d_model)
        # 当前 PyTorch 版本的 TransformerEncoderLayer 不支持独立的 attn_dropout，
        # 注意力 dropout 由通用 dropout 参数控制（attn 层内部使用同一个 dropout 值）
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead,
            dim_feedforward=ffn_mult * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, output_size)
        self.init_weights()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # 1. 输入投影到 d_model 空间，并存为残差 identity
        identity = self.input_proj(src)                           # (B, T, d_model)
        # 2. 加位置编码 + dropout
        x = self.pos_encoder(identity) * math.sqrt(self.d_model)  # (B, T, d_model)
        x = self.dropout(x)
        # 3. Transformer encoder
        x = self.transformer_encoder(x)                            # (B, T, d_model)
        # 4. 线性解码到 output_size
        decoder_out = self.decoder(x)                              # (B, T, output_size)
        # 5. 逐特征门控，防止污染物被压平
        scaled = decoder_out * self.feature_scale                 # (B, T, output_size)
        # 6. 残差 shortcut：sigmoid(logit) ∈ (0, 1) 软约束
        w = torch.sigmoid(self.residual_weight)
        identity_out = self.residual_proj(identity)               # (B, T, output_size)
        return scaled + w * identity_out                           # (B, T, output_size)
