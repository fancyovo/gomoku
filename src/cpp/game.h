#pragma once
#include "board.h"
#include <vector>
#include <cstdint>

struct BlockResult {
    int end_step;  // 0..31 if finished, -1 if still ongoing after 32 steps
    int result;    // 1=black_win, 2=white_win, 3=draw (valid only if end_step >= 0)
};

struct GamePool {
    int N;
    std::vector<GomokuBoard> boards;
    std::vector<bool> finished;   // true if game ended this wave

    GamePool(int pool_size);

    // Reset all boards for a new wave.
    void reset_all();

    // Execute 32 actions per game. Parallel via OpenMP.
    // indices:     (batch,)    — which pool slots to process
    // actions_32:  (batch*32)  — flat array, row-major: game0_step0..step31, game1_step0..
    // out_results: (batch*2)   — interleaved: [end_step0, result0, end_step1, result1, ...]
    void execute_block(
        const int* indices,
        const int* actions_32,
        int batch_size,
        int* out_results
    );

    // Get active (not finished) indices.
    std::vector<int> active_indices() const;
    int active_count() const;

    // Get move history for a board.
    std::vector<int> get_moves(int pool_idx) const;
};
