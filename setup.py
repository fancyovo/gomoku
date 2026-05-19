from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "gomoku_cpp",
        [
            "src/cpp/board.cpp",
            "src/cpp/game.cpp",
            "src/cpp/mcts.cpp",
            "src/cpp/bindings.cpp",
        ],
        include_dirs=["src/cpp"],
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        cxx_std=17,
    ),
]

setup(
    name="gomoku_transformer",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
