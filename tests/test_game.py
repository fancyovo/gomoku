import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_cpp_import():
    try:
        import gomoku_cpp
        print(f"  [PASS] gomoku_cpp imported (board_size={gomoku_cpp.board_size()})")
    except ImportError:
        print("  [SKIP] gomoku_cpp not built — run 'pip install -e .' first")


def test_board_basic():
    try:
        import gomoku_cpp
        b = gomoku_cpp.Board()
        assert b.result == 0
        assert b.num_moves == 0
        assert b.current_player == 0

        r = b.play_move(0)
        assert r == 0
        assert b.num_moves == 1
        assert b.current_player == 1
        print("  [PASS] test_board_basic")
    except ImportError:
        print("  [SKIP] gomoku_cpp not built")


def test_illegal_move():
    try:
        import gomoku_cpp
        b = gomoku_cpp.Board()
        b.play_move(0)
        r = b.play_move(0)  # occupied
        assert r == -1
        assert b.result == 2  # white loses (played illegal)
        print("  [PASS] test_illegal_move")
    except ImportError:
        print("  [SKIP] gomoku_cpp not built")


def test_game_manager():
    try:
        import gomoku_cpp
        mgr = gomoku_cpp.GameManager(8, seed=123)
        assert mgr.active_count == 8

        # Step all games
        actions = [i for i in range(8)]  # each plays a different first move
        finished = mgr.step(mgr.active_indices, actions)
        assert len(finished) == 0  # no game ends after 1 move
        assert mgr.active_count == 8
        print("  [PASS] test_game_manager")
    except ImportError:
        print("  [SKIP] gomoku_cpp not built")


if __name__ == "__main__":
    print("Running game tests...")
    test_cpp_import()
    test_board_basic()
    test_illegal_move()
    test_game_manager()
    print("Done!")
