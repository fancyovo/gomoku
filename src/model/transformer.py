import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .config import ModelConfig
from .embeddings import ActionEmbedding


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms).to(dtype) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.dropout = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)

        q = q.transpose(1, 2)  # (b, n_heads, s, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(b, s, d)
        return self.proj(out)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GomokuTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = ActionEmbedding(config.n_positions, config.d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.norm_f = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.n_positions, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, positions: torch.Tensor, players: torch.Tensor):
        """
        Args:
            positions: (batch, seq_len)
            players:   (batch, seq_len)
        Returns:
            logits: (batch, seq_len, n_positions)
        """
        x = self.embedding(positions, players)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.head(x)

    @torch.inference_mode()
    def get_logits(self, positions: torch.Tensor, players: torch.Tensor):
        """FP16 inference: return logits at the last position only."""
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = self.forward(positions, players)
        return out[:, -1, :].float()

    @torch.inference_mode()
    def sample_actions(self, positions: torch.Tensor, players: torch.Tensor):
        """Sample one action per batch element from the policy distribution."""
        logits = self.get_logits(positions, players)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
