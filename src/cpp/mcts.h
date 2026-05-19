#pragma once
#include <vector>
#include <random>
#include <cstdint>
#include <cstring>

constexpr int MCTS_ACTIONS = 225;

struct MCTSEdge {
    int action = -1;
    int child_idx = -1;
    int N = 0;
    float W = 0.0f;
    float P = 0.0f;
    float Q() const { return N > 0 ? W / N : 0.0f; }
};

struct MCTSNode {
    int parent = -1;
    int action_leading = -1;
    std::vector<MCTSEdge> edges;
    int N_total = 0;
    float V = 0.0f;
    bool expanded = false;
    int virtual_losses = 0;
    int player = 0;
    uint64_t occupied[4] = {0, 0, 0, 0};
    uint64_t p0_stones[4] = {0, 0, 0, 0};
    uint64_t p1_stones[4] = {0, 0, 0, 0};
    bool terminal = false;
    float terminal_value = 0.0f;

    int find_edge(int action) const {
        for (size_t i = 0; i < edges.size(); i++)
            if (edges[i].action == action) return (int)i;
        return -1;
    }

    bool is_occupied(int pos) const {
        return (occupied[pos >> 6] >> (pos & 63)) & 1;
    }

    void copy_full(const MCTSNode& src) {
        for (int i = 0; i < 4; i++) {
            occupied[i] = src.occupied[i];
            p0_stones[i] = src.p0_stones[i];
            p1_stones[i] = src.p1_stones[i];
        }
    }

    void add_stone(int pos, int plr) {
        int chunk = pos >> 6; uint64_t bit = 1ULL << (pos & 63);
        occupied[chunk] |= bit;
        if (plr == 0) p0_stones[chunk] |= bit;
        else p1_stones[chunk] |= bit;
    }
};

struct MCTSTree {
    std::vector<MCTSNode> nodes;
    int game_idx;
    std::mt19937 rng;

    std::vector<int> sel_nodes;
    std::vector<int> sel_edges;
    std::vector<int> sel_actions;

    MCTSTree(int gid, uint64_t seed) : game_idx(gid), rng(seed) {
        nodes.reserve(2048);
    }

    void init_root(int start_player, const uint64_t p0[4], const uint64_t p1[4]);
    int create_child(int parent_idx, int action, int child_player);
    int select_leaf(float c_puct);
};

class MCTSManager {
public:
    int num_games;
    std::vector<MCTSTree> trees;
    std::vector<int> finished;
    std::vector<int> game_players;     // [num_games] current player at root (0=black,1=white)
    std::vector<uint64_t> game_occ;    // [num_games*4] occupied bitboards per game

    float c_puct = 1.0f;
    float dirichlet_eps = 0.25f;
    float dirichlet_alpha = 0.03f;
    int leaves_per_game = 16;  // M: multi-leaf selection

    struct SelectResult {
        int n_total;                              // total leaves = active_games * M
        std::vector<int> game_indices;            // [n_total] which game each leaf belongs to
        std::vector<int> all_actions;             // flat: all path actions concatenated
        std::vector<int> leaf_offsets;            // [n_total+1] start offset in all_actions
        std::vector<int> leaf_lengths;            // [n_total] path length per leaf
        int max_path_len = 0;

        // Dense outputs for zero-copy Python→GPU transfer:
        // pos_dense: (n_total, max_path_len) padded positions
        // plr_dense: (n_total, max_path_len) padded players
        // occ_dense: (n_total, 225) occupied mask per leaf (game+path)
        // valid_mask: (n_total,) bool — which leaves have non-empty paths
        std::vector<int> pos_dense;
        std::vector<int> plr_dense;
        std::vector<uint8_t> occ_dense;  // bool per byte
        std::vector<uint8_t> valid_mask; // bool per byte

        // deprecated
        std::vector<std::vector<int>> path_actions;
    };

    MCTSManager(int n_games, uint64_t seed_base);

    void init_roots(const bool* p0_flat, const bool* p1_flat, const int* first_player);

    // Select M leaves per game. Each leaf uses virtual loss to diversify.
    SelectResult select_all();

    // Expand roots directly (before any select_all, when roots are unexpanded).
    // game_indices: (n_games,) — which games to expand
    void expand_roots(const int* game_indices, int n_games,
                      const float* policy_priors, const float* values);

    // Expand non-terminal leaves and backup. All leaves processed in one call.
    // leaf_indices: (n_leaves,) — global leaf index from select_all result (0..n_total-1)
    void expand_and_backup(const int* leaf_indices, int n_leaves,
                           const float* policy_priors, const float* values);

    void get_root_policies(float* out) const;

    void apply_move(int game_idx, int action, const bool* new_p0, const bool* new_p1);

    void reset_game(int game_idx);
    std::vector<int> active_indices() const;

private:
    struct GameSelectState {
        int game_idx = -1;
        std::vector<int> path_nodes;
        std::vector<int> path_edges;
        std::vector<int> path_actions;
        int leaf_node_idx = -1;
    };
    // Per-leaf state: sel_states_[leaf_idx] for each of the M*G leaves
    std::vector<GameSelectState> sel_states_;
};
