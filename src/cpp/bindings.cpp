#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "board.h"
#include "game.h"

namespace py = pybind11;

PYBIND11_MODULE(gomoku_cpp, m) {
    m.doc() = "Gomoku board engine with OpenMP";

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

            // out: (batch, 3) — [end_step, result, end_reason] per game
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

    m.def("board_size", []() { return BOARD_SIZE; });
    m.def("board_cells", []() { return BOARD_CELLS; });
}
