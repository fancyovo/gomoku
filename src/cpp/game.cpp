#include "game.h"
#include <algorithm>

GameManager::GameManager(int num_games, int seed) : rng(seed), games_per_step(num_games) {
    boards.resize(num_games);
    for (int i = 0; i < num_games; i++) {
        boards[i].reset();
        active_indices.push_back(i);
    }
    total_games_started = num_games;
}

int GameManager::replenish() {
    int needed = games_per_step - static_cast<int>(active_indices.size());
    if (needed <= 0) return 0;

    int added = 0;
    for (int i = 0; i < static_cast<int>(boards.size()) && added < needed; i++) {
        if (boards[i].result != 0) {
            boards[i].reset();
            active_indices.push_back(i);
            added++;
            total_games_started++;
        }
    }

    // If we ran out of slots, expand
    while (added < needed) {
        boards.emplace_back();
        active_indices.push_back(static_cast<int>(boards.size()) - 1);
        added++;
        total_games_started++;
    }

    return added;
}

std::vector<int> GameManager::get_action_sequence(int pool_idx) const {
    const auto& b = boards[pool_idx];
    std::vector<int> seq;
    seq.reserve(b.num_moves);
    for (int i = 0; i < b.num_moves; i++) {
        seq.push_back(b.move_history[i]);
    }
    return seq;
}

std::vector<int> GameManager::step(const std::vector<int>& pool_indices,
                                    const std::vector<int>& actions) {
    std::vector<int> finished;

    for (size_t i = 0; i < pool_indices.size(); i++) {
        int idx = pool_indices[i];
        int result = boards[idx].play_move(actions[i]);

        if (result != 0) {
            finished.push_back(idx);
            // Remove from active
            auto it = std::find(active_indices.begin(), active_indices.end(), idx);
            if (it != active_indices.end()) {
                active_indices.erase(it);
            }
        }
    }

    return finished;
}
