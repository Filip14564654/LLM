import torch
import torch.nn as nn
import math
from typing import Optional

# Základní implementace autoregresivního Transformeru ve stylu minGPT/NanoGPT

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, heads, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.scale = embed_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, T, self.heads, C // self.heads).transpose(1, 2), qkv)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.out(out)
        out = self.proj_drop(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, ff_dim, dropout: float = 0.0):
        super().__init__()
        self.attn = SelfAttention(embed_dim, heads, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, ff_dim=256, num_layers=4,
                 max_len=512, dropout: float = 0.0, positional_encoding: str = "learned"):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = positional_encoding
        if positional_encoding == "learned":
            self.pos_emb = nn.Embedding(max_len, embed_dim)
        else:
            pe = self._build_sinusoidal(max_len, embed_dim)
            self.register_buffer("pos_emb_table", pe, persistent=False)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout=dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _build_sinusoidal(self, length: int, dim: int) -> torch.Tensor:
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)

        token_emb = self.token_emb(idx)
        if self.positional_encoding == "learned":
            pos_emb = self.pos_emb(pos).unsqueeze(0)
        else:
            pos_emb = self.pos_emb_table[pos].unsqueeze(0)

        x = token_emb + pos_emb
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
