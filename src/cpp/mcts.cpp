#include "mcts.h"
#include "board.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <omp.h>

// Win check using per-player bitboards (same algorithm as GomokuBoard)
static int check_win_func(const uint64_t* stones, int last_pos) {
    for (int d = 0; d < 4; d++) {
        int step = DIR_STEPS[d];
        int max_p = max_steps(last_pos, d, 0);
        int max_n = max_steps(last_pos, d, 1);
        int count = 1;
        for (int i = 1, p = last_pos + step; i <= max_p; i++, p += step) {
            if (!((stones[p >> 6] >> (p & 63)) & 1)) break;
            count++;
        }
        for (int i = 1, p = last_pos - step; i <= max_n; i++, p -= step) {
            if (!((stones[p >> 6] >> (p & 63)) & 1)) break;
            count++;
        }
        if (count >= 5) return 1;
    }
    return 0;
}

// ─── MCTSTree ───────────────────────────────────────────────────

void MCTSTree::init_root(int start_player, const uint64_t p0[4], const uint64_t p1[4]) {
    nodes.clear();
    MCTSNode root;
    root.player = start_player;
    for (int i = 0; i < 4; i++) {
        root.occupied[i] = p0[i] | p1[i];
        root.p0_stones[i] = p0[i];
        root.p1_stones[i] = p1[i];
    }
    nodes.push_back(root);
}

int MCTSTree::create_child(int parent_idx, int action, int child_player) {
    int child_idx = (int)nodes.size();
    MCTSNode child;
    child.parent = parent_idx;
    child.action_leading = action;
    child.player = child_player;
    // Inherit stones from parent, add the action for parent's player
    child.copy_full(nodes[parent_idx]);
    int mover = 1 - child_player;  // parent's player just moved
    child.add_stone(action, mover);
    // Check for terminal (5-in-a-row)
    const uint64_t* stones = (mover == 0) ? child.p0_stones : child.p1_stones;
    if (check_win_func(stones, action)) {
        child.terminal = true;
        child.terminal_value = -1.0f;  // from leaf player's perspective: mover won, leaf lost
    }
    nodes.push_back(child);
    return child_idx;
}

int MCTSTree::select_leaf(float c_puct) {
    sel_nodes.clear();
    sel_edges.clear();
    sel_actions.clear();

    int node_idx = 0;
    MCTSNode* node = &nodes[node_idx];

    while (node->expanded && !node->edges.empty() && !node->terminal) {
        node->virtual_losses++;
        node->N_total++;

        int best_e = -1;
        float best_score = -1e10f;
        float sqrt_N = std::sqrt((float)node->N_total);

        for (int i = 0; i < (int)node->edges.size(); i++) {
            const MCTSEdge& e = node->edges[i];
            float Q = e.Q();
            float U = c_puct * e.P * sqrt_N / (1.0f + e.N);
            float score = Q + U;
            if (score > best_score) {
                best_score = score;
                best_e = i;
            }
        }

        if (best_e < 0) break;

        MCTSEdge& edge = node->edges[best_e];
        sel_nodes.push_back(node_idx);
        sel_edges.push_back(best_e);
        sel_actions.push_back(edge.action);

        if (edge.child_idx < 0) {
            int child_player = 1 - node->player;
            edge.child_idx = create_child(node_idx, edge.action, child_player);
            node_idx = edge.child_idx;
            node = &nodes[node_idx];
            break;
        }

        node_idx = edge.child_idx;
        node = &nodes[node_idx];
    }

    return node_idx;
}

// ─── MCTSManager ─────────────────────────────────────────────────

MCTSManager::MCTSManager(int n_games, uint64_t seed_base)
    : num_games(n_games), finished(n_games, 0),
      game_players(n_games, 0), game_occ(n_games * 4, 0)
{
    trees.reserve(n_games);
    for (int i = 0; i < n_games; i++) {
        trees.emplace_back(i, seed_base + i * 131071);
    }
    sel_states_.resize(n_games * leaves_per_game);
}

// Helper: convert bool[225] to uint64_t[4]
static void bool_to_occ(const bool* src, uint64_t* dst) {
    dst[0] = dst[1] = dst[2] = dst[3] = 0;
    for (int p = 0; p < MCTS_ACTIONS; p++) {
        if (src[p]) dst[p >> 6] |= (1ULL << (p & 63));
    }
}

// Helper: check if position is occupied in uint64_t[4]
static bool occ_test(const uint64_t* occ, int pos) {
    return (occ[pos >> 6] >> (pos & 63)) & 1;
}

void MCTSManager::init_roots(const bool* p0_flat, const bool* p1_flat, const int* first_player) {
    for (int i = 0; i < num_games; i++) {
        uint64_t p0[4], p1[4];
        bool_to_occ(p0_flat + i * MCTS_ACTIONS, p0);
        bool_to_occ(p1_flat + i * MCTS_ACTIONS, p1);
        trees[i].init_root(first_player[i], p0, p1);
        finished[i] = 0;
        game_players[i] = first_player[i];
        for (int j = 0; j < 4; j++) game_occ[i * 4 + j] = p0[j] | p1[j];
    }
}

std::vector<int> MCTSManager::active_indices() const {
    std::vector<int> out;
    out.reserve(num_games);
    for (int i = 0; i < num_games; i++)
        if (!finished[i]) out.push_back(i);
    return out;
}

MCTSManager::SelectResult MCTSManager::select_all() {
    auto active = active_indices();
    int n_active = (int)active.size();
    int M = leaves_per_game;
    int n_total = n_active * M;

    SelectResult res;
    res.n_total = n_total;
    res.game_indices.resize(n_total);
    res.leaf_lengths.resize(n_total);
    res.max_path_len = 0;

    sel_states_.assign(n_total, GameSelectState{});

    // First pass: compute total actions and per-leaf lengths (parallel)
    #pragma omp parallel for
    for (int ai = 0; ai < n_active; ai++) {
        int g = active[ai];
        if (finished[g]) continue;

        MCTSTree& tree = trees[g];
        int base_leaf = ai * M;
        for (int m = 0; m < M; m++) {
            int leaf_idx = tree.select_leaf(c_puct);
            int li = base_leaf + m;
            GameSelectState& st = sel_states_[li];
            st.game_idx = g;
            st.path_nodes = tree.sel_nodes;
            st.path_edges = tree.sel_edges;
            st.path_actions = tree.sel_actions;
            st.leaf_node_idx = leaf_idx;
            res.game_indices[li] = g;
            int plen = (int)tree.sel_actions.size();
            res.leaf_lengths[li] = plen;
            if (plen > res.max_path_len) {
                #pragma omp critical
                if (plen > res.max_path_len) res.max_path_len = plen;
            }
        }
    }

    // Build offset array and flat action array (sequential, cheap)
    res.leaf_offsets.resize(n_total + 1);
    res.leaf_offsets[0] = 0;
    for (int i = 0; i < n_total; i++) {
        res.leaf_offsets[i + 1] = res.leaf_offsets[i] + res.leaf_lengths[i];
    }
    int total_actions = res.leaf_offsets[n_total];
    res.all_actions.resize(total_actions);
    for (int i = 0; i < n_total; i++) {
        int g = res.game_indices[i];
        if (finished[g]) continue;
        int off = res.leaf_offsets[i];
        int len = res.leaf_lengths[i];
        const auto& acts = sel_states_[i].path_actions;
        for (int j = 0; j < len; j++) {
            res.all_actions[off + j] = acts[j];
        }
    }

    // Backward compat
    res.path_actions.resize(n_total);
    for (int i = 0; i < n_total; i++) {
        int off = res.leaf_offsets[i];
        int len = res.leaf_lengths[i];
        res.path_actions[i] = std::vector<int>(
            res.all_actions.begin() + off,
            res.all_actions.begin() + off + len);
    }

    // Build dense arrays for zero-copy Python→GPU transfer
    int max_pl = res.max_path_len;
    res.valid_mask.assign(n_total, 0);
    if (max_pl == 0) return res;

    res.pos_dense.resize(n_total * max_pl, 0);
    res.plr_dense.resize(n_total * max_pl, 0);
    res.occ_dense.resize(n_total * MCTS_ACTIONS, 0);

    for (int i = 0; i < n_total; i++) {
        int g = res.game_indices[i];
        if (finished[g]) continue;
        int len = res.leaf_lengths[i];
        if (len == 0) continue;

        res.valid_mask[i] = 1;

        // Fill pos_dense and plr_dense
        int off = res.leaf_offsets[i];
        int* pos_row = &res.pos_dense[i * max_pl];
        int* plr_row = &res.plr_dense[i * max_pl];
        for (int j = 0; j < len; j++) {
            pos_row[j] = res.all_actions[off + j];
            plr_row[j] = (game_players[g] + 1 + j) % 2;
        }

        // Fill occ_dense: copy game occupied + path actions
        uint8_t* occ_row = &res.occ_dense[i * MCTS_ACTIONS];
        const uint64_t* gocc = &game_occ[g * 4];
        for (int p = 0; p < MCTS_ACTIONS; p++) {
            occ_row[p] = occ_test(gocc, p) ? 1 : 0;
        }
        for (int j = 0; j < len; j++) {
            occ_row[pos_row[j]] = 1;
        }
    }

    return res;
}

static std::vector<float> generate_dirichlet(std::mt19937& rng, int n, float alpha) {
    std::vector<float> noise(n);
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    float total = 0.0f;
    for (int i = 0; i < n; i++) {
        noise[i] = gamma(rng);
        total += noise[i];
    }
    if (total > 0) {
        for (int i = 0; i < n; i++) noise[i] /= total;
    } else {
        noise[0] = 1.0f;
    }
    return noise;
}

void MCTSManager::expand_roots(const int* game_indices, int n_games,
                                const float* policy_priors, const float* values) {
    for (int i = 0; i < n_games; i++) {
        int g = game_indices[i];
        if (g < 0 || g >= num_games || finished[g]) continue;
        MCTSTree& tree = trees[g];
        if (tree.nodes.empty()) continue;
        MCTSNode& root = tree.nodes[0];

        const float* priors = policy_priors + i * MCTS_ACTIONS;

        std::vector<std::pair<float, int>> candidates;
        candidates.reserve(64);
        for (int a = 0; a < MCTS_ACTIONS; a++) {
            if (!root.is_occupied(a) && priors[a] > 0.001f) {
                candidates.emplace_back(priors[a], a);
            }
        }
        if (candidates.empty()) {
            int best_a = -1; float best_p = -1.0f;
            for (int a = 0; a < MCTS_ACTIONS; a++) {
                if (!root.is_occupied(a) && priors[a] > best_p) {
                    best_p = priors[a]; best_a = a;
                }
            }
            if (best_a >= 0) candidates.emplace_back(best_p, best_a);
        }

        std::vector<float> noise;
        if (dirichlet_eps > 0 && !candidates.empty()) {
            noise = generate_dirichlet(tree.rng, (int)candidates.size(), dirichlet_alpha);
        }

        root.edges.clear();
        root.edges.reserve(candidates.size());
        for (size_t j = 0; j < candidates.size(); j++) {
            MCTSEdge e;
            e.action = candidates[j].second;
            e.P = candidates[j].first;
            if (dirichlet_eps > 0 && j < noise.size()) {
                e.P = (1.0f - dirichlet_eps) * e.P + dirichlet_eps * noise[j];
            }
            e.child_idx = -1;
            root.edges.push_back(e);
        }
        root.V = values[i];
        root.expanded = !root.edges.empty();
    }
}

void MCTSManager::expand_and_backup(const int* leaf_indices, int n_leaves,
                                     const float* policy_priors, const float* values) {
    // Sequential: backup modifies shared ancestor nodes (N_total, virtual_losses)
    for (int i = 0; i < n_leaves; i++) {
        int li = leaf_indices[i];
        if (li < 0 || li >= (int)sel_states_.size()) continue;

        GameSelectState& st = sel_states_[li];
        int g = st.game_idx;
        if (g < 0 || g >= num_games || finished[g]) continue;
        if (st.leaf_node_idx < 0 || st.leaf_node_idx >= (int)trees[g].nodes.size()) continue;

        MCTSTree& tree = trees[g];
        MCTSNode& leaf = tree.nodes[st.leaf_node_idx];

        // Terminal leaf: skip NN expansion, backup with actual outcome
        if (leaf.terminal) {
            float v = leaf.terminal_value;
            for (int j = (int)st.path_nodes.size() - 1; j >= 0; j--) {
                int n_idx = st.path_nodes[j]; int e_idx = st.path_edges[j];
                if (n_idx >= (int)tree.nodes.size()) continue;
                MCTSNode& n = tree.nodes[n_idx];
                if (e_idx >= (int)n.edges.size()) continue;
                MCTSEdge& e = n.edges[e_idx];
                e.N++; e.W += v; n.N_total++;
                n.virtual_losses = std::max(0, n.virtual_losses - 1);
                v = -v;
            }
            continue;
        }

        const float* priors = policy_priors + li * MCTS_ACTIONS;

        // Filter: legal actions (not occupied) with P > threshold
        std::vector<std::pair<float, int>> candidates;
        candidates.reserve(64);
        for (int a = 0; a < MCTS_ACTIONS; a++) {
            if (!leaf.is_occupied(a) && priors[a] > 0.001f) {
                candidates.emplace_back(priors[a], a);
            }
        }

        if (candidates.empty()) {
            int best_a = -1;
            float best_p = -1.0f;
            for (int a = 0; a < MCTS_ACTIONS; a++) {
                if (!leaf.is_occupied(a) && priors[a] > best_p) {
                    best_p = priors[a]; best_a = a;
                }
            }
            if (best_a >= 0) {
                candidates.emplace_back(best_p, best_a);
            }
        }

        bool is_root = st.path_nodes.empty();
        std::vector<float> noise;
        if (is_root && dirichlet_eps > 0 && !candidates.empty()) {
            noise = generate_dirichlet(tree.rng, (int)candidates.size(), dirichlet_alpha);
        }

        leaf.edges.clear();
        leaf.edges.reserve(candidates.size());
        for (size_t i = 0; i < candidates.size(); i++) {
            MCTSEdge e;
            e.action = candidates[i].second;
            e.P = candidates[i].first;
            if (is_root && dirichlet_eps > 0 && i < noise.size()) {
                e.P = (1.0f - dirichlet_eps) * e.P + dirichlet_eps * noise[i];
            }
            e.child_idx = -1;
            e.N = 0;
            e.W = 0.0f;
            leaf.edges.push_back(e);
        }

        leaf.V = values[li];
        leaf.expanded = !leaf.edges.empty();

        // Backup
        float v = values[li];
        for (int i = (int)st.path_nodes.size() - 1; i >= 0; i--) {
            int n_idx = st.path_nodes[i];
            int e_idx = st.path_edges[i];
            if (n_idx >= (int)tree.nodes.size()) continue;
            MCTSNode& node = tree.nodes[n_idx];
            if (e_idx >= (int)node.edges.size()) continue;
            MCTSEdge& edge = node.edges[e_idx];
            edge.N++;
            edge.W += v;
            node.N_total++;
            node.virtual_losses = std::max(0, node.virtual_losses - 1);
            v = -v;
        }
    }
}

void MCTSManager::get_root_policies(float* out) const {
    std::memset(out, 0, num_games * MCTS_ACTIONS * sizeof(float));
    for (int g = 0; g < num_games; g++) {
        if (finished[g] || trees[g].nodes.empty()) continue;
        const MCTSNode& root = trees[g].nodes[0];
        float* row = out + g * MCTS_ACTIONS;

        float total = 0.0f;
        for (const auto& e : root.edges) {
            if (e.N > 0) {
                row[e.action] = (float)e.N;
                total += (float)e.N;
            }
        }

        if (total > 0) {
            for (int a = 0; a < MCTS_ACTIONS; a++) row[a] /= total;
        } else if (!root.edges.empty()) {
            float u = 1.0f / (float)root.edges.size();
            for (const auto& e : root.edges) row[e.action] = u;
        }
    }
}

void MCTSManager::apply_move(int game_idx, int action, const bool* new_p0, const bool* new_p1) {
    if (finished[game_idx]) return;
    MCTSTree& tree = trees[game_idx];
    if (tree.nodes.empty()) return;

    // Update game state with per-player stones
    game_players[game_idx] = 1 - game_players[game_idx];
    uint64_t p0[4], p1[4];
    bool_to_occ(new_p0, p0);
    bool_to_occ(new_p1, p1);
    for (int j = 0; j < 4; j++) game_occ[game_idx * 4 + j] = p0[j] | p1[j];

    MCTSNode& root = tree.nodes[0];
    int edge_idx = root.find_edge(action);

    if (edge_idx >= 0 && root.edges[edge_idx].child_idx >= 0) {
        int child_idx = root.edges[edge_idx].child_idx;
        MCTSNode new_root = tree.nodes[child_idx];
        new_root.parent = -1;
        new_root.action_leading = -1;
        for (auto& e : new_root.edges) { e.child_idx = -1; }
        // Update occupied and per-player stones from actual board
        for (int i = 0; i < 4; i++) {
            new_root.occupied[i] = p0[i] | p1[i];
            new_root.p0_stones[i] = p0[i];
            new_root.p1_stones[i] = p1[i];
        }
        tree.nodes.clear();
        tree.nodes.push_back(new_root);
    } else {
        int next_player = 1 - root.player;
        tree.nodes.clear();
        MCTSNode new_root;
        new_root.player = next_player;
        for (int i = 0; i < 4; i++) {
            new_root.occupied[i] = p0[i] | p1[i];
            new_root.p0_stones[i] = p0[i];
            new_root.p1_stones[i] = p1[i];
        }
        tree.nodes.push_back(new_root);
    }
}

void MCTSManager::reset_game(int game_idx) {
    finished[game_idx] = 1;
    trees[game_idx].nodes.clear();
}
