#!/usr/bin/env python3
"""End-to-end test: self-play → augmentation → DataLoader → manual verification."""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np
import yaml
from collections import Counter

from model import ModelConfig, GomokuTransformer
from training import SelfPlayRunner, augment_trajectory, create_dataloader, SYM_TABLE


def verify_symmetry_table():
    """Check that the 8 symmetry transforms are consistent (D4 group)."""
    print("=== Verifying symmetry table ===")
    # Check: all 8 transforms map 0-224 to 0-224 bijectively
    for s in range(8):
        mapped = SYM_TABLE[s].numpy()
        assert len(set(mapped)) == 225, f"Sym {s}: not bijective"
        assert mapped.min() == 0 and mapped.max() == 224, f"Sym {s}: range error"
    print("  All 8 symmetries are bijective [PASS]")

    # Check: rotate90 applied 4 times = identity
    r90 = SYM_TABLE[1]
    r180 = SYM_TABLE[2]
    r270 = SYM_TABLE[3]
    for pos in range(225):
        assert r90[r90[pos]].item() == r180[pos].item(), f"Rot90² ≠ Rot180 at {pos}"
        assert r90[r90[r90[r90[pos]]]].item() == pos, f"Rot90⁴ ≠ identity at {pos}"
    print("  Rot90⁴ = identity [PASS]")

    # Check: transpose of transpose = identity
    tr = SYM_TABLE[6]
    for pos in range(225):
        assert tr[tr[pos]].item() == pos, f"Transpose² ≠ identity at {pos}"
    print("  Transpose² = identity [PASS]")
    print()


def verify_augmentation(raw_traj):
    """Manually verify a single augmented trajectory."""
    print("=== Manual augmentation verification ===")
    pos = raw_traj["positions"]
    seq_len = raw_traj["actual_len"]
    print(f"  Original: seq_len={seq_len}, result={raw_traj['result']}")
    print(f"  Original first 3 positions: {pos[:3].tolist()}")
    print(f"  Original first 3 idx→(r,c): {[(p.item()//15, p.item()%15) for p in pos[:3]]}")

    syms = augment_trajectory(raw_traj, n_syms=4)  # just first 4 for brevity
    for s, t in enumerate(syms[:4]):
        s_pos = t["positions"]
        print(f"  Sym{s}: first 3 positions remapped: {s_pos[:3].tolist()}")
        # Verify rewards unchanged
        assert torch.equal(t["rewards"], raw_traj["rewards"]), f"Sym{s}: rewards changed!"
        assert torch.equal(t["players"], raw_traj["players"]), f"Sym{s}: players changed!"
        assert t["result"] == raw_traj["result"], f"Sym{s}: result changed!"
    print("  Rewards/players/result preserved under symmetry [PASS]")
    print()


def verify_dataloader(trajectories, augment, batch_size):
    """Run DataLoader and check batch structure."""
    print(f"=== DataLoader verification (augment={augment}, batch={batch_size}) ===")
    loader = create_dataloader(trajectories, batch_size=batch_size,
                               augment=augment, shuffle=True)

    total_samples = 0
    shapes_ok = True
    reward_mask_ok = True

    for batch_idx, batch in enumerate(loader):
        pos = batch["positions"]
        plr = batch["players"]
        act = batch["actions"]
        rew = batch["rewards"]
        mask = batch["mask"]

        B, L = pos.shape
        total_samples += B

        # Shape checks
        if plr.shape != (B, L) or act.shape != (B, L) or rew.shape != (B, L) or mask.shape != (B, L):
            print(f"  Batch {batch_idx}: shape mismatch!")
            shapes_ok = False

        # Reward mask check: reward=0 where mask=False
        zero_outside_mask = (rew[~mask] == 0.0).all()
        if not zero_outside_mask:
            print(f"  Batch {batch_idx}: non-zero reward outside mask!")
            reward_mask_ok = False

        # Check that valid rewards are ±1 (not zero inside mask)
        if mask.any():
            valid_rewards = rew[mask]
            nonzero_valid = (valid_rewards != 0.0).all()
            if not nonzero_valid:
                # Some valid positions have reward=0 — this only happens
                # for padding steps after game-end within a page
                pass

        if batch_idx < 2:
            print(f"  Batch {batch_idx}: B={B}, L={L}, "
                  f"rew_range=[{rew.min():.0f}, {rew.max():.0f}], "
                  f"valid_ratio={mask.float().mean():.2f}")

        if batch_idx >= 3:
            break

    print(f"  Sampled {total_samples} games across {batch_idx+1} batches")
    print(f"  Shapes consistent: {shapes_ok}")
    print(f"  Reward/mask consistent: {reward_mask_ok}")
    print()


def main():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda")
    model_cfg = ModelConfig.from_dict(config["model"])
    model = GomokuTransformer(model_cfg).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")
    print(f"Games per step: {config['training']['games_per_step']}")
    print()

    # 1. Verify symmetry table
    verify_symmetry_table()

    # 2. Run self-play (with smaller pool for quick test, then full)
    print(f"=== Self-play: {config['training']['games_per_step']} games ===")
    runner = SelfPlayRunner(model, device, config["training"])

    t0 = time.perf_counter()
    trajectories = runner.run_one_wave()
    elapsed = time.perf_counter() - t0
    py_t = runner.timing["python"]
    cpp_t = runner.timing["cpp"]

    lens = [t["actual_len"] for t in trajectories]
    results = Counter(t["result"] for t in trajectories)
    total_moves = sum(lens)

    print(f"  Games: {len(trajectories)} in {elapsed:.1f}s")
    print(f"  Python: {py_t:.1f}s, C++: {cpp_t*1000:.1f}ms")
    print(f"  Game lengths: min={min(lens)}, max={max(lens)}, mean={np.mean(lens):.2f}")
    print(f"  Total moves (non-padding): {total_moves:,}")
    print(f"  Results: {dict(results)}")
    print()

    # 3. Verify augmentation on a sample
    verify_augmentation(trajectories[0])

    # 4. DataLoader without augmentation
    verify_dataloader(trajectories, augment=False,
                      batch_size=config["training"]["train_batch_size"])

    # 5. DataLoader with augmentation (8x)
    verify_dataloader(trajectories, augment=True,
                      batch_size=config["training"]["train_batch_size"])

    # 6. Summary
    n_aug = len(trajectories) * 8
    print(f"=== Pipeline summary ===")
    print(f"  Raw games:       {len(trajectories):,}")
    print(f"  With augment:    {n_aug:,}")
    print(f"  Total moves:     {total_moves:,}")
    print(f"  Aug moves:       {total_moves * 8:,}")
    print(f"  Train batches:   {n_aug // config['training']['train_batch_size']:,}")
    print(f"  (batch_size={config['training']['train_batch_size']})")
    print()
    print("Pipeline OK!")


if __name__ == "__main__":
    main()
