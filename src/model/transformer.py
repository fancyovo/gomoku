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
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(b, s, d)
        return self.proj(out)

    def forward_decode(self, x, cache, layer_idx, indices):
        b, s, d = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.n_heads, self.d_head)
        q, k_new, v_new = qkv.unbind(dim=2)
        q = q.transpose(1, 2); k_new = k_new.transpose(1, 2); v_new = v_new.transpose(1, 2)
        k_full, v_full, new_lens = cache.write_and_get_kv(layer_idx, indices, k_new, v_new)
        max_len = new_lens.max().item()
        pos = torch.arange(max_len, device=x.device)
        mask = pos.unsqueeze(0) >= new_lens.unsqueeze(1)
        mask = mask.unsqueeze(1).unsqueeze(1)
        out = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=mask, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(b, s, d)
        return self.proj(out)

    def prefill_store(self, x, cache, layer_idx, indices):
        b, s, d = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        idx_t = torch.as_tensor(indices, device=x.device)
        cache.k[layer_idx][idx_t, :, :s, :] = k
        cache.v[layer_idx][idx_t, :, :s, :] = v
        cache.seq_lens[idx_t] = s
        out = out.transpose(1, 2).reshape(b, s, d)
        return self.proj(out)

    def prefill_extend(self, x, cache, layer_idx, indices):
        """Extend existing KV cache with d new tokens via concatenation (no cache write)."""
        b, d_new, _ = x.shape
        T_old = cache.seq_lens[indices]
        T_max = T_old.max().item()
        qkv = self.qkv(x).reshape(b, d_new, 3, self.n_heads, self.d_head)
        q, k_new, v_new = qkv.unbind(dim=2)
        q = q.transpose(1, 2); k_new = k_new.transpose(1, 2); v_new = v_new.transpose(1, 2)
        k_old = cache.k[layer_idx][indices, :, :T_max, :]
        v_old = cache.v[layer_idx][indices, :, :T_max, :]
        k_full = torch.cat([k_old, k_new], dim=2)
        v_full = torch.cat([v_old, v_new], dim=2)
        total_len = T_max + d_new
        col_idx = torch.arange(total_len, device=x.device).view(1, 1, 1, -1)
        row_r = torch.arange(d_new, device=x.device).view(1, 1, -1, 1)
        T_b = T_old.view(-1, 1, 1, 1)
        stale_old = (col_idx >= T_b) & (col_idx < T_max)
        future_new = (col_idx >= T_max) & (col_idx - T_max > row_r)
        mask = stale_old | future_new
        out = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=mask, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(b, d_new, self.d_model)
        return self.proj(out)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        gate = F.silu(self.gate_proj(x)); up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)

    def forward(self, x):
        return x + self.ffn(self.norm2(x + self.attn(self.norm1(x))))

    def forward_decode(self, x, cache, layer_idx, indices):
        x = x + self.attn.forward_decode(self.norm1(x), cache, layer_idx, indices)
        return x + self.ffn(self.norm2(x))

    def prefill_store(self, x, cache, layer_idx, indices):
        x = x + self.attn.prefill_store(self.norm1(x), cache, layer_idx, indices)
        return x + self.ffn(self.norm2(x))

    def prefill_extend(self, x, cache, layer_idx, indices):
        x = x + self.attn.prefill_extend(self.norm1(x), cache, layer_idx, indices)
        return x + self.ffn(self.norm2(x))


class GomokuTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        n_shared = getattr(config, 'n_shared', 4)
        n_policy = getattr(config, 'n_policy', 4)
        n_value = getattr(config, 'n_value', 4)

        self.has_policy = n_policy > 0
        self.has_value = n_value > 0

        self.embedding = ActionEmbedding(config.n_positions, config.d_model, config.board_size)

        # Shared backbone
        self.shared_layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(n_shared)
        ])

        # Policy branch
        if self.has_policy:
            self.policy_layers = nn.ModuleList([
                TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
                for _ in range(n_policy)
            ])
            self.policy_norm = RMSNorm(config.d_model)
            self.policy_head = nn.Linear(config.d_model, config.n_positions, bias=False)
        else:
            self.policy_layers = nn.ModuleList()
            self.policy_norm = None
            self.policy_head = None

        # Value branch
        if self.has_value:
            self.value_layers = nn.ModuleList([
                TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
                for _ in range(n_value)
            ])
            self.value_norm = RMSNorm(config.d_model)
            self.value_head = nn.Sequential(
                nn.Linear(config.d_model, config.value_head_dim, bias=False),
                nn.ReLU(),
                nn.Linear(config.value_head_dim, 2, bias=False),
            )
        else:
            self.value_layers = nn.ModuleList()
            self.value_norm = None
            self.value_head = None

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

    @staticmethod
    def _value_to_scalar(value_logits):
        probs = torch.softmax(value_logits.float(), dim=-1)
        return probs[..., 0] - probs[..., 1]

    # ── Branch layer helpers (no cache, causal attention) ──
    def _forward_branch(self, layers: nn.ModuleList, x: torch.Tensor):
        for layer in layers:
            x = layer(x)
        return x

    def _forward_branch_decode(self, layers: nn.ModuleList, x: torch.Tensor):
        """Single-token decode for branch: attention is trivial (1 token)."""
        for layer in layers:
            h = layer.norm1(x)
            qkv = layer.attn.qkv(h).reshape(h.shape[0], 1, 3, layer.attn.n_heads, layer.attn.d_head)
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
            out = out.transpose(1, 2).reshape(h.shape[0], 1, layer.attn.d_model)
            x = x + layer.attn.proj(out)
            x = x + layer.ffn(layer.norm2(x))
        return x

    # ── Public API ──
    def forward(self, positions, players):
        x = self.embedding(positions, players)
        for layer in self.shared_layers:
            x = layer(x)

        policy = None; value = None
        if self.has_policy:
            xp = self._forward_branch(self.policy_layers, x)
            xp = self.policy_norm(xp)
            policy = self.policy_head(xp)
        if self.has_value:
            xv = self._forward_branch(self.value_layers, x)
            xv = self.value_norm(xv)
            value = self.value_head(xv)
        return policy, value

    @torch.inference_mode()
    def get_logits(self, positions, players):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            policy, _ = self.forward(positions, players)
        return policy[:, -1, :].float()

    @torch.inference_mode()
    def sample_first_moves(self, batch_size, device):
        probs = torch.softmax(self.first_move_logits, dim=-1)
        return torch.multinomial(probs.unsqueeze(0).expand(batch_size, -1),
                                 num_samples=1).squeeze(-1)

    def create_cache(self, max_games, max_cache_len=None):
        if max_cache_len is None:
            max_cache_len = self.config.max_seq_len
        n_shared = len(self.shared_layers)
        if n_shared == 0:
            return None  # no shared layers, no KV cache needed
        return KVCacheManager(
            max_games=max_games, max_seq_len=max_cache_len,
            n_layers=n_shared,
            n_heads=self.config.n_heads,
            d_head=self.config.d_model // self.config.n_heads,
            device=next(self.parameters()).device,
        )

    def _run_shared(self, x, cache, indices, mode='extend'):
        for i, layer in enumerate(self.shared_layers):
            if cache is not None:
                if mode == 'store':
                    x = layer.prefill_store(x, cache, i, indices)
                elif mode == 'decode':
                    x = layer.forward_decode(x, cache, i, indices)
                else:  # extend
                    x = layer.prefill_extend(x, cache, i, indices)
            else:
                x = layer(x)
        return x

    @torch.inference_mode()
    def prefill(self, positions, players, cache, indices):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x = self.embedding(positions, players)
            x = self._run_shared(x, cache, indices, 'store')

            policy = None; value = None
            if self.has_policy:
                xp = self._forward_branch(self.policy_layers, x)
                xp = self.policy_norm(xp)
                policy = self.policy_head(xp)
            if self.has_value:
                xv = self._forward_branch(self.value_layers, x)
                xv = self.value_norm(xv)
                value = self.value_head(xv)

        last_idx = torch.tensor([p.shape[0] - 1 for p in positions], device=x.device)
        p_out = policy[torch.arange(len(indices)), last_idx].float() if policy is not None else None
        v_out = self._value_to_scalar(value[torch.arange(len(indices)), last_idx]) if value is not None else None
        return p_out, v_out

    @torch.inference_mode()
    def decode(self, positions, players, cache, indices):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pos_t = positions.unsqueeze(1); plr_t = players.unsqueeze(1)
            offset = cache.seq_lens[indices] if cache is not None else None
            x = self.embedding(pos_t, plr_t, seq_offset=offset)
            x = self._run_shared(x, cache, indices, 'decode')

            policy = None; value = None
            if self.has_policy:
                xp = self._forward_branch_decode(self.policy_layers, x)
                xp = self.policy_norm(xp)
                policy = self.policy_head(xp)
            if self.has_value:
                xv = self._forward_branch_decode(self.value_layers, x)
                xv = self.value_norm(xv)
                value = self.value_head(xv)

        if cache is not None:
            cache.advance(indices)
        p_out = policy.squeeze(1).float() if policy is not None else None
        v_out = self._value_to_scalar(value.squeeze(1)) if value is not None else None
        return p_out, v_out

    @torch.inference_mode()
    def prefill_extend(self, positions, players, cache, indices):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            offset = cache.seq_lens[indices] if cache is not None else None
            x = self.embedding(positions, players, seq_offset=offset)
            x = self._run_shared(x, cache, indices, 'extend')

            policy = None; value = None
            if self.has_policy:
                xp = self._forward_branch(self.policy_layers, x)
                xp = self.policy_norm(xp)
                policy = self.policy_head(xp)
            if self.has_value:
                xv = self._forward_branch(self.value_layers, x)
                xv = self.value_norm(xv)
                value = self.value_head(xv)

        p_out = policy.float() if policy is not None else None
        v_out = self._value_to_scalar(value) if value is not None else None
        return p_out, v_out

    @torch.inference_mode()
    def evaluate_mcts_leaves(self, positions, players, cache, indices, path_lengths):
        policy, value = self.prefill_extend(positions, players, cache, indices)
        leaf_idx = (path_lengths - 1).clamp(min=0)
        leaf_policy = policy[torch.arange(len(indices)), leaf_idx] if policy is not None else None
        leaf_value = value[torch.arange(len(indices)), leaf_idx] if value is not None else None
        return leaf_policy, leaf_value

    def load_state_dict(self, state_dict, strict=False):
        return super().load_state_dict(state_dict, strict=strict)
