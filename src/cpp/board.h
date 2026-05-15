#pragma once
#include <cstdint>
#include <vector>

constexpr int BOARD_SIZE = 15;
constexpr int BOARD_CELLS = BOARD_SIZE * BOARD_SIZE;  // 225
constexpr int MAX_MOVES = BOARD_CELLS;

struct GomokuBoard {
    uint64_t stones[2][4] = {};  // [player][chunk], 4×64=256 bits
    int move_history[MAX_MOVES];
    int num_moves = 0;
    int result = 0;  // 0=ongoing, 1=black_win, 2=white_win, 3=draw
    int current_player = 0;  // 0=black, 1=white

    void reset();

    // Returns: -1=illegal (current player loses), 0=ongoing, 1=black_win, 2=white_win
    int play_move(int pos);

    bool is_occupied(int pos) const;
    int check_win(int last_pos) const;

    // Serialize board state for Python: [2, 15, 15] numpy array
    void get_state(int* out) const;
};

// Batch stepping: apply one move to each board in parallel
std::vector<int> step_batch(
    std::vector<GomokuBoard>& boards,
    const std::vector<int>& actions
);
