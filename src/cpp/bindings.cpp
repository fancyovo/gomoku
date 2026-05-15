#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "board.h"
#include "game.h"

namespace py = pybind11;

PYBIND11_MODULE(gomoku_cpp, m) {
    m.doc() = "Gomoku board engine with OpenMP acceleration";

    py::class_<GomokuBoard>(m, "Board")
        .def(py::init<>())
        .def("reset", &GomokuBoard::reset)
        .def("play_move", &GomokuBoard::play_move)
        .def("is_occupied", &GomokuBoard::is_occupied)
        .def("check_win", &GomokuBoard::check_win)
        .def("get_state", [](const GomokuBoard& b) {
            std::vector<int> state(BOARD_CELLS);
            b.get_state(state.data());
            return state;
        })
        .def("get_moves", [](const GomokuBoard& b) {
            return std::vector<int>(b.move_history, b.move_history + b.num_moves);
        })
        .def_readonly("result", &GomokuBoard::result)
        .def_readonly("num_moves", &GomokuBoard::num_moves)
        .def_readonly("current_player", &GomokuBoard::current_player);

    py::class_<GameManager>(m, "GameManager")
        .def(py::init<int, int>(),
             py::arg("num_games"),
             py::arg("seed") = 42)
        .def("replenish", &GameManager::replenish)
        .def("active_count", &GameManager::active_count)
        .def("get_action_sequence", &GameManager::get_action_sequence)
        .def("step", &GameManager::step)
        .def_property_readonly("active_indices", [](const GameManager& gm) {
            return gm.active_indices;
        })
        .def_property_readonly("total_games_started", [](const GameManager& gm) {
            return gm.total_games_started;
        });

    m.def("step_batch", &step_batch);
    m.def("board_size", []() { return BOARD_SIZE; });
    m.def("board_cells", []() { return BOARD_CELLS; });
}
