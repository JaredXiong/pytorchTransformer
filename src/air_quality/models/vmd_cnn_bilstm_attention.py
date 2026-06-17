"""
VMD-CNN-BiLSTM-Attention 模型(双 head 版)

数据流:input → 多尺度 CNN → BiLSTM → Attention → [pretrain_head | finetune_head]

注：VMD 分解在数据预处理阶段离线完成（参见 vmd_features.apply_vmd_to_aqi），
模型 forward 接收已经拼接好的 (B, T, input_size) 输入。

双 head 设计:
- pretrain_head: Linear(bilstm_hidden*2, vmd_K) → 预测 K 个 IMF 分量
- finetune_head: Sequential(Linear + ReLU + Dropout + Linear) → 预测 7 维污染物
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel


class TemporalAttention(nn.Module):
    """时间步注意力层"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.v(torch.tanh(self.W(x)))
        weights = torch.softmax(scores, dim=1)
        return x * weights


class VMDCNNBILSTMAttentionModel(BaseModel):
    """VMD-CNN-BiLSTM-Attention 混合模型(双 head)"""

    def __init__(
        self,
        input_size: int,
        output_size: int = 7,
        vmd_K: int = 4,
        cnn_filters: int = 64,
        bilstm_hidden: int = 128,
        bilstm_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(input_size, output_size, dropout)
        self.vmd_K = vmd_K

        # === 共享 backbone ===
        self.input_proj = nn.Linear(input_size, cnn_filters)
        self.conv3 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(cnn_filters * 3)
        self.relu = nn.ReLU()
        self.bilstm = nn.LSTM(
            cnn_filters * 3,
            bilstm_hidden,
            bilstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if bilstm_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(bilstm_hidden * 2)

        # === 预训练 head ===
        self.pretrain_head = nn.Linear(bilstm_hidden * 2, vmd_K)

        # === 微调 head ===
        self.finetune_head = nn.Sequential(
            nn.Linear(bilstm_hidden * 2, bilstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bilstm_hidden, output_size),
        )

        self.init_weights()

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h_t = h.transpose(1, 2)
        h3 = self.conv3(h_t)
        h5 = self.conv5(h_t)
        h7 = self.conv7(h_t)
        h_cat = torch.cat([h3, h5, h7], dim=1)
        h_cat = self.bn(h_cat)
        h_cat = self.relu(h_cat)
        h_cat = F.adaptive_max_pool1d(h_cat, h_t.size(2))
        h_seq = h_cat.transpose(1, 2)
        lstm_out, _ = self.bilstm(h_seq)
        attended = self.attention(lstm_out)
        return attended

    def forward(self, x: torch.Tensor, mode: str = 'finetune') -> torch.Tensor:
        attended = self._backbone(x)
        if mode == 'pretrain':
            return self.pretrain_head(attended)
        elif mode == 'finetune':
            return self.finetune_head(attended)
        else:
            raise ValueError(f"mode 必须是 'pretrain' 或 'finetune',当前: {mode}")
