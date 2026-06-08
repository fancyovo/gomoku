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
        policy, value = model(positions, players)

    assert policy.shape == (2, 10, 225), f"Expected policy (2, 10, 225), got {policy.shape}"
    assert value.shape == (2, 10, 2), f"Expected value (2, 10, 2), got {value.shape}"
    print("  [PASS] test_model_forward")


def test_causal_mask():
    config = ModelConfig(d_model=128, n_layers=4, n_heads=4, d_ff=256)
    model = GomokuTransformer(config)
    model.eval()

    positions = torch.randint(0, 225, (1, 5))
    players = torch.randint(0, 2, (1, 5))

    # Forward with 3 tokens → output at position 2
    with torch.no_grad():
        policy_short, _ = model(positions[:, :3], players[:, :3])
        logits_short = policy_short[0, -1, :]

    # Forward with 5 tokens → output at position 2 should be same
    with torch.no_grad():
        policy_long, _ = model(positions, players)
        logits_long = policy_long[0, 2, :]

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


def test_sample_first_moves():
    config = ModelConfig(d_model=128, n_layers=4, n_heads=4, d_ff=256)
    model = GomokuTransformer(config)
    model.eval()

    actions = model.sample_first_moves(8, torch.device("cpu"))
    assert actions.shape == (8,), f"Expected (8,), got {actions.shape}"
    assert all(0 <= a < 225 for a in actions)
    print("  [PASS] test_sample_first_moves")


def test_param_count():
    config = ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256)
    model = GomokuTransformer(config)
    n = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n:,}")
    assert 2_000_000 <= n <= 2_300_000, f"Expected ~2.15M params, got {n:,}"
    print("  [PASS] test_param_count")


if __name__ == "__main__":
    print("Running model tests...")
    test_model_forward()
    test_causal_mask()
    test_get_logits()
    test_sample_first_moves()
    test_param_count()
    print("All tests passed!")
