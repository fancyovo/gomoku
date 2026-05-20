#!/usr/bin/env python3
"""Test: verify board terminal detection works correctly."""
import numpy as np
import gomoku_cpp

# Test 1: Create a 5-in-a-row manually and check detection
board = gomoku_cpp.Board()
# Play 4 stones for black horizontally
moves_4 = [0, 1, 2, 3]  # top row
for a in moves_4:
    r = board.play_move(a)
    print(f"Black plays {a}: result={r}")
    if r != 0:
        board.display()
# 5th stone should win
r = board.play_move(4)
print(f"Black plays 4 (5-in-a-row): result={r} (expect 1=black win)")

print()

# Test 2: GamePool with a forced win sequence
pool = gomoku_cpp.GamePool(2)
pool.reset_all()

# Game 0: black builds 5-in-a-row at 0,1,2,3,4
for a in [0, 15, 1, 16, 2, 17, 3, 18]:
    r = gomoku_cpp.step(pool, 0, a)
    print(f"Game 0 move {a}: result={r}")
r = gomoku_cpp.step(pool, 0, 4)
print(f"Game 0 move 4 (5th in row): result={r} (expect 1=black win)")

print()

# Test 3: Does pool properly reject moves on occupied cells?
r = gomoku_cpp.step(pool, 0, 0)
print(f"Game 0 move on occupied cell 0: result={r} (expect -1=illegal)")

print()

# Test 4: Verify that a full game actually terminates
pool2 = gomoku_cpp.GamePool(4)
pool2.reset_all()
# Play random moves until all games end
np.random.seed(42)
results = [0, 0, 0, 0]
lengths = [0, 0, 0, 0]
for move_idx in range(225):
    for g in range(4):
        if results[g] == 0:
            # Pick random unoccupied position
            # Not easy to track, just try random
            for _ in range(100):
                a = np.random.randint(0, 225)
                r = gomoku_cpp.step(pool2, g, a)
                if r != -1:
                    lengths[g] += 1
                    if r != 0:
                        results[g] = r
                    break
    if all(r != 0 for r in results):
        break

for g in range(4):
    outcome = {1: "Black", 2: "White", 3: "Draw"}[results[g]]
    print(f"Game {g}: {outcome} wins, {lengths[g]} moves")
