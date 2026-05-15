import torch
import torch.nn as nn
import math

from .config import ModelConfig
from .embeddings import ActionEmbedding


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        b, s, d = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each: (b, s, n_heads, d_head)

        # (b, n_heads, s, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1)) * scale  # (b, n_heads, s, s)

        if mask is not None:
            # mask: (s, s) causal → broadcast to (b, n_heads, s, s)
            attn = attn.masked_fill(mask[:s, :s] == 0, float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (b, n_heads, s, d_head)
        out = out.transpose(1, 2).reshape(b, s, d)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
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
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.n_positions, bias=False)

        # Causal mask buffer
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
                     .view(1, 1, config.max_seq_len, config.max_seq_len),
        )

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
            x = layer(x, self.causal_mask)
        x = self.ln_f(x)
        return self.head(x)

    @torch.inference_mode()
    def get_logits(self, positions: torch.Tensor, players: torch.Tensor):
        """FP16 inference: return logits at the last position only."""
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = self.forward(positions, players)
        return out[:, -1, :].float()  # (batch, n_positions)

    @torch.inference_mode()
    def sample_actions(self, positions: torch.Tensor, players: torch.Tensor):
        """Sample one action per batch element from the policy distribution."""
        logits = self.get_logits(positions, players)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
