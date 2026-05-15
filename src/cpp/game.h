#pragma once
#include "board.h"
#include <vector>
#include <random>

struct GameManager {
    std::vector<GomokuBoard> boards;
    std::vector<int> active_indices;  // indices of ongoing games
    std::mt19937 rng;
    int games_per_step;
    int total_games_started = 0;

    GameManager(int num_games, int seed = 42);

    // Start new games to fill pool up to target size
    int replenish();

    // Get current active board count
    int active_count() const { return static_cast<int>(active_indices.size()); }

    // Get the action sequence for an active game (by pool index)
    std::vector<int> get_action_sequence(int pool_idx) const;

    // Compact finished games: returns list of (pool_idx) that just finished
    std::vector<int> step(const std::vector<int>& pool_indices,
                          const std::vector<int>& actions);
};
