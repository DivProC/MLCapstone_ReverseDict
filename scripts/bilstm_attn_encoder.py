"""
BiLSTM encoder with a multi-head attention pooling layer, for the reverse
dictionary task.

Drop-in compatible with the existing pipeline: `BiLSTMAttnEncoder.forward`
has the exact same signature and output contract as the original
`BiLSTMEncoder` in bilstm_encoder.py —

    forward(token_ids: LongTensor[B, T], lengths: LongTensor[B])
        -> L2-normalised FloatTensor[B, output_dim]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-head additive (Bahdanau-style) self-attention pooling over a
    sequence of BiLSTM outputs.

    
    """

    def __init__(self, input_dim: int, attn_dim: int = 128, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.attn_dim = attn_dim

        # Shared key projection (tanh nonlinearity = classic Bahdanau attention).
        self.key_proj = nn.Linear(input_dim, attn_dim)
        # One learned query vector per head.
        self.query = nn.Parameter(torch.randn(num_heads, attn_dim) * (attn_dim ** -0.5))
        # Combine the concatenated per-head contexts back to input_dim.
        self.out_proj = nn.Linear(input_dim * num_heads, input_dim)

    def forward(self, outputs: torch.Tensor, mask: torch.Tensor):
        """
        outputs : (B, T, D) BiLSTM per-timestep outputs
        mask    : (B, T) bool, True at valid (non-padding) positions

        Returns
        -------
        context      : (B, D) attention-pooled representation
        attn_weights : (B, num_heads, T) softmax attention weights, for
                       optional inspection/visualisation.
        """
        keys = torch.tanh(self.key_proj(outputs))                       # (B, T, A)
        scores = torch.einsum("bta,ha->bht", keys, self.query)          # (B, heads, T)
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)                    # (B, heads, T)

        context = torch.einsum("bht,btd->bhd", attn_weights, outputs)   # (B, heads, D)
        context = context.reshape(context.size(0), -1)                  # (B, heads*D)
        context = self.out_proj(context)                                # (B, D)
        return context, attn_weights


class BiLSTMAttnEncoder(nn.Module):
    """
    BiLSTM encoder + multi-head attention pooling head.

    Same constructor/forward contract as BiLSTMEncoder, plus two extra
    optional knobs (attn_dim, num_heads) that default to sensible values.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
        attn_dim: int = 128,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        lstm_out_dim = hidden_dim * 2  # bidirectional
        self.attn_pool = MultiHeadAttentionPooling(lstm_out_dim, attn_dim=attn_dim, num_heads=num_heads)

        # final_hidden (2H) concatenated with attention context (2H) -> 4H
        self.projection = nn.Sequential(
            nn.Linear(lstm_out_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _make_mask(token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """(B, T) bool mask, True at positions < that sample's length."""
        B, T = token_ids.shape
        arange = torch.arange(T, device=token_ids.device).unsqueeze(0).expand(B, T)
        return arange < lengths.to(token_ids.device).unsqueeze(1)

    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
        return_attention: bool = False,
    ):
        mask = self._make_mask(token_ids, lengths)                      # (B, T)

        x = self.dropout(self.embedding(token_ids))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (hidden, _) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=token_ids.size(1)
        )                                                                # (B, T, 2H)
        outputs = self.dropout(outputs)

        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)        # (B, 2H)
        context, attn_weights = self.attn_pool(outputs, mask)            # (B, 2H)

        combined = self.dropout(torch.cat([final_hidden, context], dim=1))  # (B, 4H)
        embedding = F.normalize(self.projection(combined), dim=1)

        if return_attention:
            return embedding, attn_weights
        return embedding


class PredictorMLP(nn.Module):
    """Learned projection from query embedding space to target embedding space.
    Identical to the original — included here so this file is self-contained."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 512, output_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


