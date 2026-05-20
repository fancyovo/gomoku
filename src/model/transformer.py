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

        self.k = [
            torch.zeros(max_games, n_heads, max_seq_len, d_head,
                        dtype=torch.bfloat16, device=device)
            for _ in range(n_layers)
        ]
        self.v = [
            torch.zeros(max_games, n_heads, max_seq_len, d_head,
                        dtype=torch.bfloat16, device=device)
            for _ in range(n_layers)
        ]
        self.seq_lens = torch.zeros(max_games, dtype=torch.long, device=device)

    def get_mask(self, indices: torch.Tensor):
        lens = self.seq_lens[indices]
        max_len = lens.max().item()
        if max_len == 0:
            return None
        pos = torch.arange(max_len, device=self.device)
        mask = pos.unsqueeze(0) >= lens.unsqueeze(1)
        mask = mask.unsqueeze(1).unsqueeze(1)
        return mask

    def write_kv(self, layer_idx: int, indices: torch.Tensor,
                 k_new: torch.Tensor, v_new: torch.Tensor):
        pos = self.seq_lens[indices]
        self.k[layer_idx][indices, :, pos, :] = k_new.squeeze(2)
        self.v[layer_idx][indices, :, pos, :] = v_new.squeeze(2)

    def get_kv(self, layer_idx: int, indices: torch.Tensor):
        lens = self.seq_lens[indices]
        max_len = lens.max().item()
        if max_len == 0:
            return None, None
        k = self.k[layer_idx][indices, :, :max_len, :]
        v = self.v[layer_idx][indices, :, :max_len, :]
        return k, v

    def write_and_get_kv(self, layer_idx: int, indices: torch.Tensor,
                         k_new: torch.Tensor, v_new: torch.Tensor):
        lens = self.seq_lens[indices]
        self.k[layer_idx][indices, :, lens, :] = k_new.squeeze(2)
        self.v[layer_idx][indices, :, lens, :] = v_new.squeeze(2)
        new_lens = lens + 1
        max_len = new_lens.max().item()
        k = self.k[layer_idx][indices, :, :max_len, :]
        v = self.v[layer_idx][indices, :, :max_len, :]
        return k, v, new_lens

    def advance(self, indices: torch.Tensor):
        self.seq_lens[indices] += 1

    def reset_game(self, pool_idx: int):
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
        b, s, d = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.n_heads, self.d_head)
        q, k_new, v_new = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k_new = k_new.transpose(1, 2)
        v_new = v_new.transpose(1, 2)

        k_full, v_full, new_lens = cache.write_and_get_kv(
            layer_idx, indices, k_new, v_new
        )

        max_len = new_lens.max().item()
        pos = torch.arange(max_len, device=x.device)
        mask = pos.unsqueeze(0) >= new_lens.unsqueeze(1)
        mask = mask.unsqueeze(1).unsqueeze(1)

        out = F.scaled_dot_product_attention(
            q, k_full, v_full,
            attn_mask=mask,
            dropout_p=0.0,
        )
        out = out.transpose(1, 2).reshape(b, s, d)
        return self.proj(out)

    def prefill_store(self, x: torch.Tensor, cache: KVCacheManager,
                      layer_idx: int, indices: torch.Tensor):
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

        idx_t = torch.as_tensor(indices, device=x.device)
        cache.k[layer_idx][idx_t, :, :s, :] = k
        cache.v[layer_idx][idx_t, :, :s, :] = v
        cache.seq_lens[idx_t] = s

        out = out.transpose(1, 2).reshape(b, s, d)
        return self.proj(out)

    def prefill_extend(self, x: torch.Tensor, cache: KVCacheManager,
                       layer_idx: int, indices: torch.Tensor):
        """Extend existing KV cache with d new tokens via concatenation (no cache write).

        Builds full K,V by concatenating cached prefix with new tokens.
        No for-loop, no cache mutation — pure tensor ops.
        """
        b, d, _ = x.shape
        T_old = cache.seq_lens[indices]  # (b,)
        T_max = T_old.max().item()

        qkv = self.qkv(x).reshape(b, d, 3, self.n_heads, self.d_head)
        q, k_new, v_new = qkv.unbind(dim=2)
        q = q.transpose(1, 2)       # (b, n_heads, d, d_head)
        k_new = k_new.transpose(1, 2)
        v_new = v_new.transpose(1, 2)

        # Read cached K,V (positions 0..T_max-1) and concat with new (VECTORIZED)
        k_old = cache.k[layer_idx][indices, :, :T_max, :]
        v_old = cache.v[layer_idx][indices, :, :T_max, :]
        k_full = torch.cat([k_old, k_new], dim=2)  # (b, n_heads, T_max+d, d_head)
        v_full = torch.cat([v_old, v_new], dim=2)

        # Build mask (vectorized, no Python loops):
        # For game b with prefix length T[b], row r (0<=r<d) computes new pos T_max+r.
        # Valid attention targets:
        #   col < T[b]                    → valid old positions
        #   col >= T_max and col-T_max<=r → causal within new block
        # Masked (stale):
        #   T[b] <= col < T_max           → stale old cache from other games
        #   col >= T_max and col-T_max>r  → future new
        total_len = T_max + d
        col_idx = torch.arange(total_len, device=x.device).view(1, 1, 1, -1)  # (1,1,1,total)
        row_r   = torch.arange(d, device=x.device).view(1, 1, -1, 1)          # (1,1,d,1)
        T_b     = T_old.view(-1, 1, 1, 1)  # (b,1,1,1)

        stale_old  = (col_idx >= T_b) & (col_idx < T_max)   # (b,1,d,total)
        future_new = (col_idx >= T_max) & (col_idx - T_max > row_r)  # (b,1,d,total)
        mask = stale_old | future_new  # True = blocked

        out = F.scaled_dot_product_attention(
            q, k_full, v_full,
            attn_mask=mask,
            dropout_p=0.0,
        )
        out = out.transpose(1, 2).reshape(b, d, self.d_model)
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

    def prefill_extend(self, x: torch.Tensor, cache: KVCacheManager,
                       layer_idx: int, indices: torch.Tensor):
        x = x + self.attn.prefill_extend(self.norm1(x), cache, layer_idx, indices)
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
        self.policy_head = nn.Linear(config.d_model, config.n_positions, bias=False)
        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, config.value_head_dim, bias=False),
            nn.ReLU(),
            nn.Linear(config.value_head_dim, 1, bias=False),
            nn.Tanh(),
        )

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
        x = self.embedding(positions, players)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        policy = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return policy, value

    @torch.inference_mode()
    def get_logits(self, positions: torch.Tensor, players: torch.Tensor):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            policy, _ = self.forward(positions, players)
        return policy[:, -1, :].float()

    @torch.inference_mode()
    def sample_first_moves(self, batch_size: int, device: torch.device):
        probs = torch.softmax(self.first_move_logits, dim=-1)
        return torch.multinomial(probs.unsqueeze(0).expand(batch_size, -1),
                                 num_samples=1).squeeze(-1)

    @torch.inference_mode()
    def sample_actions(self, positions: torch.Tensor, players: torch.Tensor):
        logits = self.get_logits(positions, players)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def create_cache(self, max_games: int, max_cache_len: int | None = None) -> KVCacheManager:
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
        """Prefill: process initial sequences, populate KV cache, return policy+value."""
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x = self.embedding(positions, players)
            for i, layer in enumerate(self.layers):
                x = layer.prefill_store(x, cache, i, indices)
            x = self.norm_f(x)
            policy = self.policy_head(x)
            value = self.value_head(x).squeeze(-1)
        last_idx = torch.tensor([p.shape[0] - 1 for p in positions], device=x.device)
        return (policy[torch.arange(len(indices)), last_idx].float(),
                value[torch.arange(len(indices)), last_idx].float())

    @torch.inference_mode()
    def decode(self, positions: torch.Tensor, players: torch.Tensor,
               cache: KVCacheManager, indices: torch.Tensor):
        """Decode one token per game using KV cache. Returns (policy_logits, values)."""
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pos_t = positions.unsqueeze(1)
            plr_t = players.unsqueeze(1)
            offset = cache.seq_lens[indices]
            x = self.embedding(pos_t, plr_t, seq_offset=offset)
            for i, layer in enumerate(self.layers):
                x = layer.forward_decode(x, cache, i, indices)
            x = self.norm_f(x)
            policy = self.policy_head(x)
            value = self.value_head(x).squeeze(-1).squeeze(-1)
        cache.advance(indices)
        return policy.squeeze(1).float(), value.float()

    @torch.inference_mode()
    def prefill_extend(self, positions: torch.Tensor, players: torch.Tensor,
                       cache: KVCacheManager, indices: torch.Tensor):
        """Extend KV cache with d new tokens (MCTS path). seq_lens unchanged.

        Args:
            positions: (b, d) — new tokens to add
            players:   (b, d) — player ids
            cache:     KV cache with seq_lens = game length T
            indices:   (b,) — pool indices

        Returns:
            policy: (b, d, n_positions) — policy logits at each of the d positions
            value:  (b, d) — value at each of the d positions
        """
        # Validate at function boundary
        assert positions.max().item() < self.config.n_positions, \
            f"prefill_extend: position OOB max={positions.max().item()}, n_pos={self.config.n_positions}"
        assert positions.min().item() >= 0, \
            f"prefill_extend: negative position min={positions.min().item()}"

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            offset = cache.seq_lens[indices]  # (b,) — game length T
            x = self.embedding(positions, players, seq_offset=offset)
            for i, layer in enumerate(self.layers):
                x = layer.prefill_extend(x, cache, i, indices)
            x = self.norm_f(x)
            policy = self.policy_head(x)       # (b, d, n_positions)
            value = self.value_head(x).squeeze(-1)  # (b, d)
        return policy.float(), value.float()

    @torch.inference_mode()
    def evaluate_mcts_leaves(self, positions: torch.Tensor, players: torch.Tensor,
                             cache: KVCacheManager, indices: torch.Tensor,
                             path_lengths: torch.Tensor):
        """Batch-evaluate MCTS leaf nodes via KV cache extension.

        Args:
            positions:    (b, d_max) — padded new tokens (root-to-leaf path)
            players:      (b, d_max)
            cache:        root KV cache with seq_lens = game length T (unchanged)
            indices:      (b,)
            path_lengths: (b,) — actual path length d_i per game (1-indexed)

        Returns:
            policy_logits: (b, n_positions) at leaf position
            values:        (b,) at leaf position
        """
        policy, value = self.prefill_extend(positions, players, cache, indices)
        leaf_idx = (path_lengths - 1).clamp(min=0)  # (b,)
        leaf_policy = policy[torch.arange(len(indices)), leaf_idx]
        leaf_value = value[torch.arange(len(indices)), leaf_idx]
        # For non-root leaves (path_length > 1): model outputs value from mover's
        # perspective (plr_dense encodes the mover). Backup expects from leaf player's
        # perspective (= opponent of mover). Flip sign.
        # Root (path_length == 1): value is already from root player's perspective. No flip.
        is_leaf = path_lengths > 1
        if is_leaf.any():
            leaf_value = torch.where(is_leaf, -leaf_value, leaf_value)
        return leaf_policy, leaf_value

    def load_state_dict(self, state_dict, strict=False):
        """Load with backward compat for old single-head checkpoints."""
        return super().load_state_dict(state_dict, strict=strict)
