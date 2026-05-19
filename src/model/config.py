from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    d_model: int = 128
    n_layers: int = 16
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    max_seq_len: int = 512
    board_size: int = 15
    value_head_dim: int = 64

    @property
    def n_positions(self) -> int:
        return self.board_size * self.board_size  # 225

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
