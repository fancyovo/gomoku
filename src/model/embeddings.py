import torch
import torch.nn as nn


class ActionEmbedding(nn.Module):
    """Fuses position and player into a single latent vector via MLP."""

    def __init__(self, n_positions: int, d_model: int):
        super().__init__()
        self.pos_emb = nn.Embedding(n_positions, d_model)
        self.plr_emb = nn.Embedding(2, d_model)
        self.fusion = nn.Linear(2 * d_model, d_model)
        self.pos_encoding = nn.Embedding(512, d_model)  # learned positional encoding

    def forward(self, positions: torch.Tensor, players: torch.Tensor,
                seq_offset: torch.Tensor | None = None):
        """
        Args:
            positions: (batch, seq_len) int64 tensor of position indices
            players:   (batch, seq_len) int64 tensor of player ids (0 or 1)
            seq_offset: (batch,) — position offset for each sample (used in decode)
                        When provided, pos_encoding uses offset + arange(seq_len).
        Returns:
            (batch, seq_len, d_model) tensor
        """
        b, s = positions.shape
        seq_idx = torch.arange(s, device=positions.device).unsqueeze(0).expand(b, -1)

        if seq_offset is not None:
            seq_idx = seq_idx + seq_offset.unsqueeze(1)

        pos = self.pos_emb(positions)       # (b, s, d_model)
        plr = self.plr_emb(players)          # (b, s, d_model)
        token = self.fusion(torch.cat([pos, plr], dim=-1))  # (b, s, d_model)
        pe = self.pos_encoding(seq_idx)      # (b, s, d_model)

        return token + pe
