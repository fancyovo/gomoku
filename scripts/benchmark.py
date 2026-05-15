#!/usr/bin/env python3
"""Find max inference batch size and estimate 1M-game throughput."""

import sys
import os
import time
import torch
import argparse
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import ModelConfig, GomokuTransformer


def find_max_batch(config, seq_len=256):
    """Binary search for the largest power-of-2 batch that fits in GPU memory."""
    device = torch.device("cuda")
    print(f"Model: d_model={config.d_model}, n_layers={config.n_layers}, "
          f"n_heads={config.n_heads}, d_ff={config.d_ff}")
    print(f"Test seq_len: {seq_len}\n")

    model = GomokuTransformer(config).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")

    valid = 32
    bs = 32
    while bs <= 16384:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            positions = torch.randint(0, 225, (bs, seq_len), dtype=torch.long, device=device)
            players = torch.randint(0, 2, (bs, seq_len), dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(positions, players)
            torch.cuda.synchronize()
            mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  batch={bs:5d}  OK  peak_mem={mem:.2f} GB")
            valid = bs
            del positions, players
            bs *= 2
        except torch.cuda.OutOfMemoryError:
            print(f"  batch={bs:5d}  OOM")
            torch.cuda.empty_cache()
            break

    print(f"\nMax batch size: {valid}")
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return valid


def benchmark_throughput(config, batch_size, seq_len=256, warmup=30, measure=100):
    """Measure throughput with a fresh model."""
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = GomokuTransformer(config).to(device).eval()

    positions = torch.randint(0, 225, (batch_size, seq_len), dtype=torch.long, device=device)
    players = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.long, device=device)

    # Warmup
    print(f"Warming up ({warmup} steps)...")
    for i in range(warmup):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            _ = model(positions, players)
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  Peak memory: {peak:.2f} GB / {total:.1f} GB ({peak/total*100:.1f}%)")

    # Measure
    print(f"Measuring ({measure} steps)...")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for i in range(measure):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            _ = model(positions, players)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms_per_batch = (elapsed / measure) * 1000
    decisions_per_sec = (batch_size * measure) / elapsed

    return ms_per_batch, decisions_per_sec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--measure_steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--games", type=int, default=1_000_000,
                        help="Number of games for throughput estimate")
    args = parser.parse_args()

    config = ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256)

    # Step 1: Find max batch
    print("=" * 60)
    print("Step 1: Finding max batch size (power of 2)")
    print("=" * 60)
    max_batch = find_max_batch(config, args.seq_len)

    # Step 2: Benchmark with fresh model
    print()
    print("=" * 60)
    print(f"Step 2: Throughput at batch={max_batch}, seq_len={args.seq_len}")
    print("=" * 60)
    ms_per_batch, decisions_per_sec = benchmark_throughput(
        config, max_batch, args.seq_len, args.warmup, args.measure_steps
    )

    # Step 3: Estimate 1M games
    GAMES = args.games
    STEPS = args.seq_len
    total_decisions = GAMES * STEPS
    num_batches = total_decisions / max_batch
    total_seconds = total_decisions / decisions_per_sec

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Batch size:          {max_batch}")
    print(f"  Seq length (steps):  {STEPS}")
    print(f"  Time per batch:      {ms_per_batch:.2f} ms")
    print(f"  Decisions/sec:       {decisions_per_sec:,.0f}")
    print()
    print(f"{GAMES:,} games estimate:")
    print(f"  Total decisions:     {total_decisions:,.0f}")
    print(f"  Forward passes:      {num_batches:,.0f}")
    print(f"  Total time:          {total_seconds:.1f} sec = {total_seconds/60:.1f} min = {total_seconds/3600:.2f} h")


if __name__ == "__main__":
    main()
