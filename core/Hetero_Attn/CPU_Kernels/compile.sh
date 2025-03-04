#!/bin/bash

# # Set environment variable for library paths
export LD_LIBRARY_PATH=/home/tairan/miniconda3/envs/archer_env/lib/python3.11/site-packages/torch/lib:/usr/local/cuda-12.2/lib64:/usr/local/cuda-12.2/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# Get pybind11 include path
PYBIND11_INCLUDE=$(python3 -c "import pybind11; print(pybind11.get_include())")

# Compilation command
g++ -std=c++20 -Ofast -fipa-pta -march=native -mtune=native -ffast-math -fno-unsafe-math-optimizations -fprefetch-loop-arrays -fopenmp -pthread -D_GLIBCXX_USE_CXX11_ABI=0 \
    -o test_kernel unit_test.cpp grouped_query_attention_cpu_avx2_omp.cpp \
    -I/home/tairan/miniconda3/envs/archer_env/lib/python3.11/site-packages/torch/include \
    -I/home/tairan/miniconda3/envs/archer_env/lib/python3.11/site-packages/torch/include/torch/csrc/api/include \
    -I/usr/local/cuda-12.2/include \
    -I/home/tairan/miniconda3/envs/archer_env/include/python3.11 \
    -I/home/tairan/andrewxu313/Profile \
    -I${PYBIND11_INCLUDE} \
    -L/home/tairan/miniconda3/envs/archer_env/lib/python3.11/site-packages/torch/lib \
    -L/usr/local/cuda-12.2/lib64 \
    -L/usr/local/cuda-12.2/targets/x86_64-linux/lib \
    -L/home/tairan/miniconda3/envs/archer_env/lib \
    -Wl,-rpath,/home/tairan/miniconda3/envs/archer_env/lib/python3.11/site-packages/torch/lib \
    -Wl,-rpath,/usr/local/cuda-12.2/lib64 \
    -Wl,-rpath,/usr/local/cuda-12.2/targets/x86_64-linux/lib \
    -Wl,-rpath,/home/tairan/miniconda3/envs/archer_env/lib \
    -ltorch_python -ltorch -ltorch_cpu -lc10 -lpython3.11 -lnuma -lcudart -lcufile \
    -lpthread
