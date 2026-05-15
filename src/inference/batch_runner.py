import torch
import torch.nn as nn


class BatchRunner:
    """High-throughput FP16 batch inference for self-play."""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def infer(self, positions: torch.Tensor, players: torch.Tensor):
        """
        Args:
            positions: (batch, seq_len) padded
            players:   (batch, seq_len) padded
        Returns:
            probs: (batch, n_positions) softmax probabilities at last position
        """
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = self.model(positions.to(self.device), players.to(self.device))
        logits = logits[:, -1, :].float()  # last position only
        return torch.softmax(logits, dim=-1)

    @torch.inference_mode()
    def sample(self, positions: torch.Tensor, players: torch.Tensor):
        """Sample actions from policy. Returns (batch,) tensor of action indices."""
        probs = self.infer(positions, players)
        return torch.multinomial(probs, num_samples=1).squeeze(-1).cpu()
