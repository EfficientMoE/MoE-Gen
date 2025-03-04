// clang-format off
/* ----------------------------------------------------------------------------  *
 *  MoE-Gen                                                                      *
 *  copyright (c) EfficientMoE team 2025                                             *
 *                                                                               *
 *  licensed under the apache license, version 2.0 (the "license");              *
 *  you may not use this file except in compliance with the license.             *
 *                                                                               *
 *  you may obtain a copy of the license at                                      *
 *                                                                               *
 *                  http://www.apache.org/licenses/license-2.0                   *
 *                                                                               *
 *  unless required by applicable law or agreed to in writing, software          *
 *  distributed under the license is distributed on an "as is" basis,            *
 *  without warranties or conditions of any kind, either express or implied.     *
 *  see the license for the specific language governing permissions and          *
 *  limitations under the license.                                               *
 * ---------------------------------------------------------------------------- */
// clang-format on

/*
        Including infrastructure functions.
        - Model Config Parser
        - Engine Config Parser
        - cudaError Check
        - logger init
        - pinned memory allocation
*/
#pragma once
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <torch/cuda.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "data_structures.h"

namespace py = pybind11;
inline void throwOnCudaError(cudaError_t error, const char* file, int line,
                             const char* function, const char* call) {
    if (error != cudaSuccess) {
        std::stringstream ss;
        ss << "CUDA error " << error << " at " << file << ":" << line
           << " in function " << function << ": " << cudaGetErrorString(error)
           << "\nCall: " << call;
        throw std::runtime_error(ss.str());
    }
};

#define CUDA_CHECK(call) \
    throwOnCudaError(call, __FILE__, __LINE__, __FUNCTION__, #call)

struct sync_event {
    std::atomic<int> is_set;
    cudaEvent_t event;
    sync_event() : is_set(0) { CUDA_CHECK(cudaEventCreate(&event)); }
    ~sync_event() { cudaEventDestroy(event); }
    // Delete copy constructor and copy assignment operator
    sync_event(const sync_event&) = delete;
    sync_event& operator=(const sync_event&) = delete;
};

EngineConfig parse_engine_config(const py::object& engine_config);
ModelConfig parse_model_config(const py::object& model_config);
std::shared_ptr<spdlog::logger> init_logger(const std::string& log_level,
                                            const std::string& logger_name);

template <typename T>
constexpr T ceil_div(T numerator, T denominator) {
    return (numerator + denominator - 1) / denominator;
}
