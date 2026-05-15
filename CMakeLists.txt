cmake_minimum_required(VERSION 3.14)
project(gomoku_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimisation flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fopenmp")

# OpenMP
find_package(OpenMP REQUIRED)

# pybind11
find_package(pybind11 REQUIRED)

add_library(gomoku_cpp MODULE
    src/cpp/board.cpp
    src/cpp/game.cpp
    src/cpp/bindings.cpp
)

target_include_directories(gomoku_cpp PRIVATE src/cpp)
target_link_libraries(gomoku_cpp PRIVATE OpenMP::OpenMP_CXX pybind11::headers)

set_target_properties(gomoku_cpp PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)
