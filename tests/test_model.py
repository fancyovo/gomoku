import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from model import ModelConfig, GomokuTransformer


def test_model_forward():
    config = ModelConfig(d_model=128, n_layers=4, n_heads=4, d_ff=256)
    model = GomokuTransformer(config)
    model.eval()

    positions = torch.randint(0, 225, (2, 10))
    players = torch.randint(0, 2, (2, 10))

    with torch.no_grad():
        out = model(positions, players)

    assert out.shape == (2, 10, 225), f"Expected (2, 10, 225), got {out.shape}"
    print("  [PASS] test_model_forward")


def test_causal_mask():
    config = ModelConfig(d_model=128, n_layers=4, n_heads=4, d_ff=256)
    model = GomokuTransformer(config)
    model.eval()

    positions = torch.randint(0, 225, (1, 5))
    players = torch.randint(0, 2, (1, 5))

    # Forward with 3 tokens → output at position 2
    with torch.no_grad():
        out_short = model(positions[:, :3], players[:, :3])
        logits_short = out_short[0, -1, :]

    # Forward with 5 tokens → output at position 2 should be same
    with torch.no_grad():
        out_long = model(positions, players)
        logits_long = out_long[0, 2, :]

    diff = (logits_short - logits_long).abs().max().item()
    assert diff < 1e-5, f"Causal mask violated: diff={diff}"
    print("  [PASS] test_causal_mask")


def test_get_logits():
    config = ModelConfig(d_model=128, n_layers=4, n_heads=4, d_ff=256)
    model = GomokuTransformer(config)
    model.eval()

    positions = torch.randint(0, 225, (4, 8))
    players = torch.randint(0, 2, (4, 8))

    logits = model.get_logits(positions, players)
    assert logits.shape == (4, 225), f"Expected (4, 225), got {logits.shape}"
    print("  [PASS] test_get_logits")


def test_sample_actions():
    config = ModelConfig(d_model=128, n_layers=4, n_heads=4, d_ff=256)
    model = GomokuTransformer(config)
    model.eval()

    positions = torch.randint(0, 225, (8, 3))
    players = torch.randint(0, 2, (8, 3))

    actions = model.sample_actions(positions, players)
    assert actions.shape == (8,), f"Expected (8,), got {actions.shape}"
    assert all(0 <= a < 225 for a in actions)
    print("  [PASS] test_sample_actions")


def test_param_count():
    config = ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256)
    model = GomokuTransformer(config)
    n = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n:,}")
    assert 1_500_000 <= n <= 2_500_000, f"Expected ~2M params, got {n:,}"
    print("  [PASS] test_param_count")


if __name__ == "__main__":
    print("Running model tests...")
    test_model_forward()
    test_causal_mask()
    test_get_logits()
    test_sample_actions()
    test_param_count()
    print("All tests passed!")
