import math
import torch
import torch.nn as nn


class ActionEmbedding(nn.Module):
    """Fuses position, player, and spatial coordinate into a single latent vector via MLP."""

    def __init__(self, n_positions: int, d_model: int, board_size: int = 15):
        super().__init__()
        self.board_size = board_size
        self.pos_emb = nn.Embedding(n_positions, d_model)
        self.plr_emb = nn.Embedding(2, d_model)
        self.coord_proj = nn.Linear(4, d_model)  # sin/cos(x), sin/cos(y) → d_model
        self.fusion = nn.Linear(3 * d_model, d_model)  # [pos, plr, coord] → d_model
        self.pos_encoding = nn.Embedding(512, d_model)

    def forward(self, positions: torch.Tensor, players: torch.Tensor,
                seq_offset: torch.Tensor | None = None):
        """
        Args:
            positions: (batch, seq_len) int64 tensor of position indices
            players:   (batch, seq_len) int64 tensor of player ids (0 or 1)
            seq_offset: (batch,) — position offset for each sample (used in decode)
        Returns:
            (batch, seq_len, d_model) tensor
        """
        b, s = positions.shape
        device = positions.device
        dtype = self.pos_emb.weight.dtype
        seq_idx = torch.arange(s, device=device).unsqueeze(0).expand(b, -1)

        if seq_offset is not None:
            seq_idx = seq_idx + seq_offset.unsqueeze(1)

        pos = self.pos_emb(positions)       # (b, s, d_model)
        plr = self.plr_emb(players)          # (b, s, d_model)

        # Spatial coordinate encoding: 4D Fourier features
        sz = self.board_size
        x = positions.float() % sz          # (b, s)
        y = positions.float() // sz         # (b, s)
        pi = math.pi
        coord = torch.stack([
            torch.sin(pi * x / sz),
            torch.cos(pi * x / sz),
            torch.sin(pi * y / sz),
            torch.cos(pi * y / sz),
        ], dim=-1).to(dtype)                # (b, s, 4)
        coord = self.coord_proj(coord)      # (b, s, d_model)

        token = self.fusion(torch.cat([pos, plr, coord], dim=-1))  # (b, s, d_model)
        pe = self.pos_encoding(seq_idx)

        return token + pe
