#pragma once
#include <cstdint>
#include <cstring>
#include <algorithm>

constexpr int BOARD_SIZE = 15;
constexpr int BOARD_CELLS = BOARD_SIZE * BOARD_SIZE;  // 225

// Direction step offsets (linear index): →, ↓, ↘, ↙
constexpr int DIR_STEPS[4] = {1, BOARD_SIZE, BOARD_SIZE + 1, BOARD_SIZE - 1};

// Precomputed: max steps in each of 4 directions before hitting board edge.
// [pos][dir][0=positive_way, 1=negative_way]
// Generated at compile time via constexpr.
inline constexpr int max_steps(int pos, int dir, int way) {
    int r = pos / BOARD_SIZE;
    int c = pos % BOARD_SIZE;
    switch (dir) {
        case 0: return way == 0 ? (BOARD_SIZE - 1) - c : c;                    // horizontal
        case 1: return way == 0 ? (BOARD_SIZE - 1) - r : r;                    // vertical
        case 2: return way == 0 ? std::min((BOARD_SIZE - 1) - r, (BOARD_SIZE - 1) - c)
                                : std::min(r, c);
        case 3: return way == 0 ? std::min((BOARD_SIZE - 1) - r, c)
                                : std::min(r, (BOARD_SIZE - 1) - c);
        default: return 0;
    }
}

struct GomokuBoard {
    uint64_t stones[2][4] = {};   // bitboards: [player][chunk], 4 chunks × 64 = 256 bits
    uint64_t occupied[4] = {};    // stones[0] | stones[1], for fast single-read is_occupied
    int move_history[BOARD_CELLS];
    int num_moves = 0;
    int result = 0;              // 0=ongoing, 1=black_win, 2=white_win, 3=draw
    int current_player = 0;      // 0=black, 1=white

    void reset() {
        std::memset(stones, 0, sizeof(stones));
        std::memset(occupied, 0, sizeof(occupied));
        num_moves = 0;
        result = 0;
        current_player = 0;
    }

    inline bool is_occupied(int pos) const {
        return (occupied[pos >> 6] >> (pos & 63)) & 1;
    }

    inline void set_stone(int player, int pos) {
        uint64_t bit = 1ULL << (pos & 63);
        int chunk = pos >> 6;
        stones[player][chunk] |= bit;
        occupied[chunk] |= bit;
    }

    // Returns: -1=illegal, 0=ongoing, 1=black_win, 2=white_win, 3=draw
    inline int play_move(int pos) {
        if (pos < 0 || pos >= BOARD_CELLS || is_occupied(pos)) {
            result = (current_player == 0) ? 2 : 1;
            return -1;
        }
        set_stone(current_player, pos);
        move_history[num_moves++] = pos;

        int w = check_win(pos);
        if (w > 0) { result = w; return w; }
        if (num_moves >= BOARD_CELLS) { result = 3; return 3; }
        current_player ^= 1;
        return 0;
    }

    inline int check_win(int last_pos) const {
        int player = (num_moves & 1) ^ 1;  // who just moved
        const uint64_t* s = stones[player];

        for (int d = 0; d < 4; d++) {
            int step = DIR_STEPS[d];
            int max_p = max_steps(last_pos, d, 0);
            int max_n = max_steps(last_pos, d, 1);
            int count = 1;

            for (int i = 1, p = last_pos + step; i <= max_p; i++, p += step) {
                if (!((s[p >> 6] >> (p & 63)) & 1)) break;
                count++;
            }
            for (int i = 1, p = last_pos - step; i <= max_n; i++, p -= step) {
                if (!((s[p >> 6] >> (p & 63)) & 1)) break;
                count++;
            }
            if (count >= 5) return player ? 2 : 1;
        }
        return 0;
    }
};
