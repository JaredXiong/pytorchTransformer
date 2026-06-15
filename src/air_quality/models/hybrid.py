"""
混合模型 - Transformer + LSTM + CNN + Cross-Attention Fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel, PositionalEncoding


class LightSelfAttention(nn.Module):
    """轻量级自注意力"""

    def __init__(self, d_model: int, nhead: int = 2, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


class CrossAttentionFusion(nn.Module):
    """交叉注意力融合层"""

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        self.t_proj = nn.Linear(d_model, d_model)
        self.l_proj = nn.Linear(d_model, d_model)
        self.c_proj = nn.Linear(d_model, d_model)

        self.cross_attn_tl = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.cross_attn_tc = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.cross_attn_lt = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.cross_attn_lc = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.cross_attn_ct = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.cross_attn_cl = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)

        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
        self.fusion_norm = nn.LayerNorm(d_model)
        self.fusion_fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, t_out: torch.Tensor, l_out: torch.Tensor, c_out: torch.Tensor) -> torch.Tensor:
        t_proj = self.t_proj(t_out)
        l_proj = self.l_proj(l_out)
        c_proj = self.c_proj(c_out)

        t_from_l, _ = self.cross_attn_tl(t_proj, l_proj, l_proj)
        t_from_c, _ = self.cross_attn_tc(t_proj, c_proj, c_proj)
        t_fused = self.fusion_norm(t_proj + t_from_l + t_from_c)

        l_from_t, _ = self.cross_attn_lt(l_proj, t_proj, t_proj)
        l_from_c, _ = self.cross_attn_lc(l_proj, c_proj, c_proj)
        l_fused = self.fusion_norm(l_proj + l_from_t + l_from_c)

        c_from_t, _ = self.cross_attn_ct(c_proj, t_proj, t_proj)
        c_from_l, _ = self.cross_attn_cl(c_proj, l_proj, l_proj)
        c_fused = self.fusion_norm(c_proj + c_from_t + c_from_l)

        weights = F.softmax(self.fusion_weights, dim=0)
        fused = weights[0] * t_fused + weights[1] * l_fused + weights[2] * c_fused
        return self.fusion_fc(fused)


class TransformerBranch(nn.Module):
    """Transformer 分支"""

    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        src = self.pos_encoder(x)
        src = F.dropout(src, p=0.2, training=self.training)
        return self.transformer_encoder(src)


class LSTMEncoderBranch(nn.Module):
    """LSTM 编码器分支"""

    def __init__(self, input_size: int = 128, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2, bidirectional: bool = True):
        super().__init__()
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
        self.fc = nn.Linear(hidden_size * self.num_directions, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.relu(self.fc(lstm_out))


class CNNEncoderBranch(nn.Module):
    """CNN 编码器分支"""

    def __init__(self, input_size: int = 128, num_filters: int = 64,
                 kernel_sizes: list = None, dropout: float = 0.2):
        super().__init__()
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
        self.fc = nn.Linear(self.total_filters, input_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        conv_outputs = [conv(x) for conv in self.convs]
        out = torch.cat(conv_outputs, dim=1)
        out = nn.AdaptiveAvgPool1d(x.size(2))(out)
        out = out.permute(0, 2, 1)
        return self.relu(self.fc(out))


class HybridModel(BaseModel):
    """混合模型：Transformer + LSTM + CNN + Cross-Attention Fusion"""

    def __init__(self, input_size: int = 9, output_size: int = 7, d_model: int = 128, nhead: int = 4, dropout: float = 0.2):
        super().__init__(input_size, output_size, dropout)
        self.d_model = d_model
        self.nhead = nhead

        self.input_proj = nn.Linear(input_size, d_model)
        self.transformer_branch = TransformerBranch(d_model, nhead, num_layers=2, dropout=dropout)
        self.lstm_branch = LSTMEncoderBranch(d_model, d_model, num_layers=2, dropout=dropout)
        self.cnn_branch = CNNEncoderBranch(d_model, num_filters=64, dropout=dropout)

        self.t_self_attn = LightSelfAttention(d_model, nhead=2, dropout=dropout)
        self.l_self_attn = LightSelfAttention(d_model, nhead=2, dropout=dropout)
        self.c_self_attn = LightSelfAttention(d_model, nhead=2, dropout=dropout)

        self.fusion = CrossAttentionFusion(d_model, nhead, dropout)

        self.output_fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size)
        )

        self.residual_weight = nn.Parameter(torch.tensor(0.3))
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.input_proj(x)
        x_proj = identity

        t_out = self.transformer_branch(x_proj)
        l_out = self.lstm_branch(x_proj)
        c_out = self.cnn_branch(x_proj)

        t_out = self.t_self_attn(t_out)
        l_out = self.l_self_attn(l_out)
        c_out = self.c_self_attn(c_out)

        fused = self.fusion(t_out, l_out, c_out)
        return self.output_fc(fused + self.residual_weight * identity)
