#include "board.h"
#include <algorithm>
#include <cstring>
#include <omp.h>

void GomokuBoard::reset() {
    std::memset(stones, 0, sizeof(stones));
    num_moves = 0;
    result = 0;
    current_player = 0;
}

bool GomokuBoard::is_occupied(int pos) const {
    int chunk = pos / 64;
    int bit = pos % 64;
    return ((stones[0][chunk] >> bit) & 1) | ((stones[1][chunk] >> bit) & 1);
}

static inline void set_bit(uint64_t* arr, int pos) {
    arr[pos / 64] |= (1ULL << (pos % 64));
}

int GomokuBoard::play_move(int pos) {
    if (pos < 0 || pos >= BOARD_CELLS || is_occupied(pos)) {
        // Illegal move: current player loses
        result = (current_player == 0) ? 2 : 1;  // opponent wins
        return -1;
    }

    set_bit(stones[current_player], pos);
    move_history[num_moves++] = pos;

    int win = check_win(pos);
    if (win > 0) {
        result = win;
        return win;
    }
    if (num_moves >= BOARD_CELLS) {
        result = 3;  // draw
        return 3;
    }

    current_player ^= 1;  // switch player
    return 0;
}

int GomokuBoard::check_win(int last_pos) const {
    int row = last_pos / BOARD_SIZE;
    int col = last_pos % BOARD_SIZE;
    int player = (num_moves % 2 == 0) ? 1 : 0;  // player who just moved
    const uint64_t* s = stones[player];

    // Direction vectors: (dr, dc) for 4 directions
    const int dirs[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};

    for (int d = 0; d < 4; d++) {
        int dr = dirs[d][0], dc = dirs[d][1];
        int count = 1;

        // Positive direction
        for (int i = 1; i < 5; i++) {
            int r = row + dr * i, c = col + dc * i;
            if (r < 0 || r >= BOARD_SIZE || c < 0 || c >= BOARD_SIZE) break;
            int p = r * BOARD_SIZE + c;
            if (!((s[p / 64] >> (p % 64)) & 1)) break;
            count++;
        }
        // Negative direction
        for (int i = 1; i < 5; i++) {
            int r = row - dr * i, c = col - dc * i;
            if (r < 0 || r >= BOARD_SIZE || c < 0 || c >= BOARD_SIZE) break;
            int p = r * BOARD_SIZE + c;
            if (!((s[p / 64] >> (p % 64)) & 1)) break;
            count++;
        }

        if (count >= 5) return (player == 0) ? 1 : 2;
    }
    return 0;
}

void GomokuBoard::get_state(int* out) const {
    for (int i = 0; i < BOARD_CELLS; i++) {
        int chunk = i / 64, bit = i % 64;
        if ((stones[0][chunk] >> bit) & 1)
            out[i] = 1;  // black
        else if ((stones[1][chunk] >> bit) & 1)
            out[i] = -1; // white
        else
            out[i] = 0;  // empty
    }
}

std::vector<int> step_batch(
    std::vector<GomokuBoard>& boards,
    const std::vector<int>& actions
) {
    int n = static_cast<int>(boards.size());
    std::vector<int> results(n);

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        results[i] = boards[i].play_move(actions[i]);
    }

    return results;
}
