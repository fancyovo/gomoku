#include "game.h"
#include <omp.h>
#include <algorithm>

GamePool::GamePool(int pool_size) : N(pool_size) {
    boards.resize(N);
    finished.resize(N, false);
}

void GamePool::reset_all() {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        boards[i].reset();
        finished[i] = false;
    }
}

void GamePool::execute_block(
    const int* indices,
    const int* actions_32,
    int batch_size,
    int* out_results
) {
    #pragma omp parallel for
    for (int i = 0; i < batch_size; i++) {
        int pool_idx = indices[i];
        int end_step = -1;
        int res = 0;

        auto& board = boards[pool_idx];
        const int* acts = actions_32 + i * 32;

        for (int step = 0; step < 32; step++) {
            int r = board.play_move(acts[step]);
            if (r != 0) {  // illegal (-1) or win (1/2) or draw (3)
                end_step = step;
                res = board.result;
                finished[pool_idx] = true;
                break;
            }
        }

        out_results[i * 2]     = end_step;
        out_results[i * 2 + 1] = res;
    }
}

std::vector<int> GamePool::active_indices() const {
    std::vector<int> out;
    out.reserve(N);
    for (int i = 0; i < N; i++)
        if (!finished[i]) out.push_back(i);
    return out;
}

int GamePool::active_count() const {
    int count = 0;
    for (int i = 0; i < N; i++)
        if (!finished[i]) count++;
    return count;
}

std::vector<int> GamePool::get_moves(int pool_idx) const {
    const auto& b = boards[pool_idx];
    return std::vector<int>(b.move_history, b.move_history + b.num_moves);
}
