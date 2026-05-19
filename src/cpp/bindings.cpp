#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "board.h"
#include "game.h"
#include "mcts.h"

namespace py = pybind11;

PYBIND11_MODULE(gomoku_cpp, m) {
    m.doc() = "Gomoku board engine with OpenMP + MCTS";

    py::class_<GomokuBoard>(m, "Board")
        .def(py::init<>())
        .def("reset", &GomokuBoard::reset)
        .def("play_move", &GomokuBoard::play_move)
        .def("is_occupied", &GomokuBoard::is_occupied)
        .def_readonly("result", &GomokuBoard::result)
        .def_readonly("num_moves", &GomokuBoard::num_moves)
        .def_readonly("current_player", &GomokuBoard::current_player)
        .def("get_moves", [](const GomokuBoard& b) {
            return std::vector<int>(b.move_history, b.move_history + b.num_moves);
        });

    py::class_<GamePool>(m, "GamePool")
        .def(py::init<int>())
        .def("reset_all", &GamePool::reset_all)
        .def("active_count", &GamePool::active_count)
        .def("active_indices", &GamePool::active_indices)
        .def("get_moves", &GamePool::get_moves)
        .def("execute_block", [](GamePool& pool,
                                  py::array_t<int> indices,
                                  py::array_t<int> actions_32) {
            auto idx_buf = indices.request();
            auto act_buf = actions_32.request();
            int batch = static_cast<int>(idx_buf.size);

            auto out = py::array_t<int>({batch, 3});
            auto out_buf = out.request();

            pool.execute_block(
                static_cast<const int*>(idx_buf.ptr),
                static_cast<const int*>(act_buf.ptr),
                batch,
                static_cast<int*>(out_buf.ptr)
            );
            return out;
        });

    // Step one action on a specific game. Returns: 0=ongoing, 1=black_win, 2=white_win, 3=draw, -1=illegal
    m.def("step", [](GamePool& pool, int game_idx, int action) -> int {
        auto& board = pool.boards[game_idx];
        int r = board.play_move(action);
        if (r != 0) pool.finished[game_idx] = 1;
        return r;
    });

    m.def("get_result", [](GamePool& pool, int game_idx) -> int {
        return pool.boards[game_idx].result;
    });

    m.def("board_size", []() { return BOARD_SIZE; });
    m.def("board_cells", []() { return BOARD_CELLS; });

    // ─── MCTS Manager ──────────────────────────────────────────

    py::class_<MCTSManager>(m, "MCTSManager")
        .def(py::init<int, uint64_t>(),
             py::arg("n_games"), py::arg("seed_base") = 42)
        .def("init_roots", [](MCTSManager& mgr,
                               py::array_t<bool> p0,
                               py::array_t<bool> p1,
                               py::array_t<int> first_player) {
            auto p0b = p0.request(); auto p1b = p1.request();
            auto plr = first_player.request();
            mgr.init_roots(static_cast<const bool*>(p0b.ptr),
                           static_cast<const bool*>(p1b.ptr),
                           static_cast<const int*>(plr.ptr));
        })
        .def("select_all", [](MCTSManager& mgr) {
            auto res = mgr.select_all();
            int max_pl = res.max_path_len;
            py::dict out;
            out["max_path_len"] = max_pl;
            out["n_total"] = res.n_total;
            if (max_pl > 0) {
                int n_total = res.n_total;
                // Dense arrays: reshape for Python
                out["pos_dense"] = py::array_t<int>({n_total, max_pl}, {max_pl * (int)sizeof(int), (int)sizeof(int)}, res.pos_dense.data());
                out["plr_dense"] = py::array_t<int>({n_total, max_pl}, {max_pl * (int)sizeof(int), (int)sizeof(int)}, res.plr_dense.data());
                out["occ_dense"] = py::array_t<uint8_t>({n_total, MCTS_ACTIONS}, {MCTS_ACTIONS * (int)sizeof(uint8_t), (int)sizeof(uint8_t)}, res.occ_dense.data());
                out["valid_mask"] = py::array_t<uint8_t>({(long)n_total}, res.valid_mask.data());
                out["leaf_lengths"] = py::array_t<int>({(long)n_total}, res.leaf_lengths.data());
                out["game_indices"] = py::array_t<int>({(long)n_total}, res.game_indices.data());
            }
            return out;
        })
        .def("expand_roots", [](MCTSManager& mgr,
                                 py::array_t<int> game_indices,
                                 py::array_t<float> policy_priors,
                                 py::array_t<float> values) {
            auto gi = game_indices.request();
            auto pp = policy_priors.request();
            auto vv = values.request();
            int n = static_cast<int>(gi.size);
            mgr.expand_roots(
                static_cast<const int*>(gi.ptr), n,
                static_cast<const float*>(pp.ptr),
                static_cast<const float*>(vv.ptr));
        })
        .def("expand_and_backup", [](MCTSManager& mgr,
                                      py::array_t<int> leaf_indices,
                                      py::array_t<float> policy_priors,
                                      py::array_t<float> values) {
            auto li = leaf_indices.request();
            auto pp = policy_priors.request();
            auto vv = values.request();
            int n = static_cast<int>(li.size);
            mgr.expand_and_backup(
                static_cast<const int*>(li.ptr), n,
                static_cast<const float*>(pp.ptr),
                static_cast<const float*>(vv.ptr));
        })
        .def_readwrite("leaves_per_game", &MCTSManager::leaves_per_game)
        .def("get_root_policies", [](MCTSManager& mgr) {
            int n = mgr.num_games;
            auto out = py::array_t<float>({(long)n, (long)MCTS_ACTIONS});
            mgr.get_root_policies(static_cast<float*>(out.request().ptr));
            return out;
        })
        .def("apply_move", [](MCTSManager& mgr, int game_idx, int action,
                               py::array_t<bool> new_p0, py::array_t<bool> new_p1) {
            mgr.apply_move(game_idx, action,
                           static_cast<const bool*>(new_p0.request().ptr),
                           static_cast<const bool*>(new_p1.request().ptr));
        })
        .def("reset_game", &MCTSManager::reset_game)
        .def("active_indices", &MCTSManager::active_indices)
        .def_readwrite("c_puct", &MCTSManager::c_puct)
        .def_readwrite("dirichlet_eps", &MCTSManager::dirichlet_eps)
        .def_readwrite("dirichlet_alpha", &MCTSManager::dirichlet_alpha)
        .def_readonly("num_games", &MCTSManager::num_games);
}
