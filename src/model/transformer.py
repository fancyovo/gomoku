import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

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


class KVCacheManager:
    """Manages per-layer KV caches for a pool of games.

    Each game has its own cache slots identified by pool index.
    Games at different sequence lengths share the same batch dimension.
    """

    def __init__(self, max_games: int, max_seq_len: int,
                 n_layers: int, n_heads: int, d_head: int, device: torch.device):
        self.max_games = max_games
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.device = device

        # (max_games, n_heads, max_seq_len, d_head) per layer, FP16
        self.k = [
            torch.zeros(max_games, n_heads, max_seq_len, d_head,
                        dtype=torch.float16, device=device)
            for _ in range(n_layers)
        ]
        self.v = [
            torch.zeros(max_games, n_heads, max_seq_len, d_head,
                        dtype=torch.float16, device=device)
            for _ in range(n_layers)
        ]
        self.seq_lens = torch.zeros(max_games, dtype=torch.long, device=device)

    def get_mask(self, indices: torch.Tensor):
        """Create boolean attention mask for active games.

        Returns (n_active, 1, 1, max_len) bool tensor.
        True = masked (do NOT attend). False = attend.
        """
        lens = self.seq_lens[indices]  # (n_active,)
        max_len = lens.max().item()
        if max_len == 0:
            return None
        pos = torch.arange(max_len, device=self.device)  # (max_len,)
        mask = pos.unsqueeze(0) >= lens.unsqueeze(1)  # (n_active, max_len), True where to mask
        mask = mask.unsqueeze(1).unsqueeze(1)  # (n_active, 1, 1, max_len)
        return mask

    def write_kv(self, layer_idx: int, indices: torch.Tensor,
                 k_new: torch.Tensor, v_new: torch.Tensor):
        """Write new K, V at current seq_lens positions (before increment).
        k_new, v_new: (n_active, n_heads, 1, d_head) — squeeze the seq dim.
        """
        pos = self.seq_lens[indices]  # (n_active,)
        self.k[layer_idx][indices, :, pos, :] = k_new.squeeze(2)
        self.v[layer_idx][indices, :, pos, :] = v_new.squeeze(2)

    def get_kv(self, layer_idx: int, indices: torch.Tensor):
        """Get full K, V caches for active games up to max(seq_len)."""
        lens = self.seq_lens[indices]
        max_len = lens.max().item()
        if max_len == 0:
            return None, None
        k = self.k[layer_idx][indices, :, :max_len, :]
        v = self.v[layer_idx][indices, :, :max_len, :]
        return k, v

    def write_and_get_kv(self, layer_idx: int, indices: torch.Tensor,
                         k_new: torch.Tensor, v_new: torch.Tensor):
        """Write new K,V at seq_lens[i], then return full cache up to seq_lens[i]+1."""
        lens = self.seq_lens[indices]  # (n_active,)
        self.k[layer_idx][indices, :, lens, :] = k_new.squeeze(2)
        self.v[layer_idx][indices, :, lens, :] = v_new.squeeze(2)
        new_lens = lens + 1
        max_len = new_lens.max().item()
        k = self.k[layer_idx][indices, :, :max_len, :]
        v = self.v[layer_idx][indices, :, :max_len, :]
        return k, v, new_lens

    def advance(self, indices: torch.Tensor):
        """Increment seq_lens after a decode step."""
        self.seq_lens[indices] += 1

    def reset_game(self, pool_idx: int):
        """Reset a single game's cache (when starting a new game)."""
        self.seq_lens[pool_idx] = 0


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
        """Standard causal forward (training / prefill)."""
        b, s, d = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(b, s, d)
        return self.proj(out)

    def forward_decode(self, x: torch.Tensor, cache: KVCacheManager,
                       layer_idx: int, indices: torch.Tensor):
        """Decode one token with KV cache.

        Args:
            x: (b, 1, d_model) — new token embeddings
            cache: KV cache manager
            layer_idx: which layer this attention belongs to
            indices: (b,) — pool indices of active games
        Returns:
            (b, 1, d_model) — attention output (new token only)
        """
        b, s, d = x.shape  # s == 1
        qkv = self.qkv(x).reshape(b, s, 3, self.n_heads, self.d_head)
        q, k_new, v_new = qkv.unbind(dim=2)
        q = q.transpose(1, 2)       # (b, n_heads, 1, d_head)
        k_new = k_new.transpose(1, 2)  # (b, n_heads, 1, d_head)
        v_new = v_new.transpose(1, 2)

        # Write new K,V and get full cache
        k_full, v_full, new_lens = cache.write_and_get_kv(
            layer_idx, indices, k_new, v_new
        )
        # k_full: (b, n_heads, max_len, d_head)

        # Build mask for variable-length sequences
        max_len = new_lens.max().item()
        pos = torch.arange(max_len, device=x.device)
        mask = pos.unsqueeze(0) >= new_lens.unsqueeze(1)  # (b, max_len), True=block
        mask = mask.unsqueeze(1).unsqueeze(1)  # (b, 1, 1, max_len)

        out = F.scaled_dot_product_attention(
            q, k_full, v_full,
            attn_mask=mask,
            dropout_p=0.0,
        )
        out = out.transpose(1, 2).reshape(b, s, d)
        return self.proj(out)

    def prefill_store(self, x: torch.Tensor, cache: KVCacheManager,
                      layer_idx: int, indices: torch.Tensor):
        """Prefill: process full sequence, store all K,V in cache.

        Args:
            x: (total_tokens, d_model) — packed tokens from all prefill games
            cache: KV cache manager
            layer_idx: which layer
            indices: list of pool indices for prefill games
        Returns:
            (total_tokens, d_model) — attention output
        """
        b, s, d = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=0.0,
        )

        # Store K, V in cache (vectorized — all games have same length s)
        idx_t = torch.as_tensor(indices, device=x.device)
        cache.k[layer_idx][idx_t, :, :s, :] = k
        cache.v[layer_idx][idx_t, :, :s, :] = v
        cache.seq_lens[idx_t] = s

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

    def forward_decode(self, x: torch.Tensor, cache: KVCacheManager,
                       layer_idx: int, indices: torch.Tensor):
        x = x + self.attn.forward_decode(self.norm1(x), cache, layer_idx, indices)
        x = x + self.ffn(self.norm2(x))
        return x

    def prefill_store(self, x: torch.Tensor, cache: KVCacheManager,
                      layer_idx: int, indices: torch.Tensor):
        x = x + self.attn.prefill_store(self.norm1(x), cache, layer_idx, indices)
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

        # Learnable initial move distribution (for seq_len=0)
        self.first_move_logits = nn.Parameter(torch.zeros(config.n_positions))

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
        """Standard forward (training)."""
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
    def sample_first_moves(self, batch_size: int, device: torch.device):
        """Sample first moves from the learnable first_move_logits distribution."""
        probs = torch.softmax(self.first_move_logits, dim=-1)
        return torch.multinomial(probs.unsqueeze(0).expand(batch_size, -1),
                                 num_samples=1).squeeze(-1)

    @torch.inference_mode()
    def sample_actions(self, positions: torch.Tensor, players: torch.Tensor):
        """Sample one action per batch element from the policy distribution."""
        logits = self.get_logits(positions, players)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def create_cache(self, max_games: int, max_cache_len: int | None = None) -> KVCacheManager:
        """Create a KV cache manager for self-play."""
        if max_cache_len is None:
            max_cache_len = self.config.max_seq_len
        return KVCacheManager(
            max_games=max_games,
            max_seq_len=max_cache_len,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            d_head=self.config.d_model // self.config.n_heads,
            device=next(self.parameters()).device,
        )

    @torch.inference_mode()
    def prefill(self, positions: torch.Tensor, players: torch.Tensor,
                cache: KVCacheManager, indices: list[int]):
        """Prefill: process initial sequences and populate KV cache.

        Args:
            positions: (b, seq_len) — padded sequences
            players:   (b, seq_len) — player ids
            cache:     KV cache manager to populate
            indices:   list of pool indices for these games
        Returns:
            logits: (b, n_positions) — logits at the last position
        """
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            x = self.embedding(positions, players)
            for i, layer in enumerate(self.layers):
                x = layer.prefill_store(x, cache, i, indices)
            x = self.norm_f(x)
            logits = self.head(x)
        return logits[torch.arange(len(indices)), [p.shape[0] - 1 for p in positions]].float()

    @torch.inference_mode()
    def decode(self, positions: torch.Tensor, players: torch.Tensor,
               cache: KVCacheManager, indices: torch.Tensor):
        """Decode one token per game using KV cache.

        Args:
            positions: (b,) — new position for each active game (1 token)
            players:   (b,) — player id for the new token
            cache:     KV cache manager (updated in-place)
            indices:   (b,) — pool indices of active games (same length as positions)
        Returns:
            logits: (b, n_positions) — next-move logits
        """
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pos_t = positions.unsqueeze(1)  # (b, 1)
            plr_t = players.unsqueeze(1)
            offset = cache.seq_lens[indices]  # (b,) — current position in game
            x = self.embedding(pos_t, plr_t, seq_offset=offset)
            for i, layer in enumerate(self.layers):
                x = layer.forward_decode(x, cache, i, indices)
            x = self.norm_f(x)
            logits = self.head(x)  # (b, 1, n_positions)
        cache.advance(indices)
        return logits.squeeze(1).float()
