#include "game.h"
#include <omp.h>
#include <algorithm>

GamePool::GamePool(int pool_size) : N(pool_size) {
    boards.resize(N);
    finished.resize(N, 0);
}

void GamePool::reset_all() {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        boards[i].reset();
        finished[i] = 0;
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
        int reason = 0;

        auto& board = boards[pool_idx];
        const int* acts = actions_32 + i * 32;

        for (int step = 0; step < 32; step++) {
            int r = board.play_move(acts[step]);
            if (r == -1) {
                end_step = step;
                res = board.result;
                reason = 2;
                finished[pool_idx] = 1;
                break;
            } else if (r == 1 || r == 2) {
                end_step = step;
                res = board.result;
                reason = 1;
                finished[pool_idx] = 1;
                break;
            } else if (r == 3) {
                end_step = step;
                res = board.result;
                reason = 3;
                finished[pool_idx] = 1;
                break;
            }
        }

        out_results[i * 3]     = end_step;
        out_results[i * 3 + 1] = res;
        out_results[i * 3 + 2] = reason;
    }
}

std::vector<int> GamePool::active_indices() const {
    std::vector<int> out;
    out.reserve(N);
    for (int i = 0; i < N; i++)
        if (finished[i] == 0) out.push_back(i);
    return out;
}

int GamePool::active_count() const {
    int count = 0;
    for (int i = 0; i < N; i++)
        if (finished[i] == 0) count++;
    return count;
}

std::vector<int> GamePool::get_moves(int pool_idx) const {
    const auto& b = boards[pool_idx];
    return std::vector<int>(b.move_history, b.move_history + b.num_moves);
}
