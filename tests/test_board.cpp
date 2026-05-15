#include <cassert>
#include <iostream>
#include "board.h"

void test_empty_board() {
    GomokuBoard b;
    assert(b.num_moves == 0);
    assert(b.result == 0);
    assert(b.current_player == 0);
    std::cout << "  [PASS] test_empty_board\n";
}

void test_play_move() {
    GomokuBoard b;
    int r = b.play_move(0);
    assert(r == 0);
    assert(b.num_moves == 1);
    assert(b.current_player == 1);
    assert(b.move_history[0] == 0);
    std::cout << "  [PASS] test_play_move\n";
}

void test_illegal_move() {
    GomokuBoard b;
    b.play_move(0);  // black at (0,0)
    int r = b.play_move(0);  // white tries same spot
    assert(r == -1);
    assert(b.result == 2);  // white loses, black wins
    std::cout << "  [PASS] test_illegal_move\n";
}

void test_horizontal_win() {
    GomokuBoard b;
    // Black: (0,0)-(0,4)
    for (int c = 0; c < 4; c++) {
        b.play_move(c);      // black at (0,c)
        b.play_move(15 + c); // white at (1,c), irrelevant
    }
    int r = b.play_move(4);  // black at (0,4) — 5 in a row
    assert(r == 1);
    assert(b.result == 1);
    std::cout << "  [PASS] test_horizontal_win\n";
}

void test_vertical_win() {
    GomokuBoard b;
    for (int r = 0; r < 4; r++) {
        b.play_move(r * 15);    // black at (r,0)
        b.play_move(r * 15 + 1); // white at (r,1)
    }
    int r = b.play_move(4 * 15); // black at (4,0) — 5 in a column
    assert(r == 1);
    std::cout << "  [PASS] test_vertical_win\n";
}

void test_diagonal_win() {
    GomokuBoard b;
    for (int i = 0; i < 4; i++) {
        b.play_move(i * 15 + i);       // black at (i,i)
        b.play_move(i * 15 + i + 1);   // white offset
    }
    int r = b.play_move(4 * 15 + 4); // black at (4,4)
    assert(r == 1);
    std::cout << "  [PASS] test_diagonal_win\n";
}

void test_is_occupied() {
    GomokuBoard b;
    assert(!b.is_occupied(0));
    b.play_move(0);
    assert(b.is_occupied(0));
    assert(!b.is_occupied(1));
    std::cout << "  [PASS] test_is_occupied\n";
}

void test_get_state() {
    GomokuBoard b;
    b.play_move(0);  // black
    b.play_move(1);  // white
    int state[BOARD_CELLS];
    b.get_state(state);
    assert(state[0] == 1);
    assert(state[1] == -1);
    assert(state[2] == 0);
    std::cout << "  [PASS] test_get_state\n";
}

int main() {
    std::cout << "Running board tests...\n";
    test_empty_board();
    test_play_move();
    test_illegal_move();
    test_horizontal_win();
    test_vertical_win();
    test_diagonal_win();
    test_is_occupied();
    test_get_state();
    std::cout << "All tests passed!\n";
    return 0;
}
