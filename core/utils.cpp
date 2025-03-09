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

#include "spdlog/spdlog.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <condition_variable>
#include <memory>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <torch/cuda.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

#include "data_structures.h"
#include "utils.h"

namespace py = pybind11;

Basic_Config parse_basic_config(const py::object& engine_config) {
    Basic_Config basic_config;
    py::object basic_config_obj = engine_config.attr("Basic_Config");
    basic_config.log_level =
        basic_config_obj.attr("log_level").cast<std::string>();
    basic_config.device = basic_config_obj.attr("device").cast<int64_t>();
    basic_config.device_torch =
        torch::Device(torch::kCUDA, basic_config.device);
    basic_config.dtype_str =
        basic_config_obj.attr("dtype_str").cast<std::string>();
    if (basic_config.dtype_str == "bfloat16") {
        basic_config.dtype_torch = torch::kBFloat16;
    } else if (basic_config.dtype_str == "float8_e4m3fn") {
        basic_config.dtype_torch = torch::kFloat8_e4m3fn;
    } else {
        basic_config.dtype_torch = torch::kFloat32;
    }
    basic_config.attn_mode = basic_config_obj.attr("attn_mode").cast<int64_t>();
    basic_config.num_threads =
        basic_config_obj.attr("num_threads").cast<int64_t>();
    basic_config.module_types =
        basic_config_obj.attr("module_types").cast<std::vector<std::string>>();
    return basic_config;
};

KV_Storage_Config parse_kv_storage_config(const py::object& engine_config) {
    KV_Storage_Config kv_storage_config;
    py::object kv_storage_config_obj = engine_config.attr("KV_Storage_Config");
    kv_storage_config.num_host_slots =
        kv_storage_config_obj.attr("num_host_slots").cast<int64_t>();
    kv_storage_config.reserved_length =
        kv_storage_config_obj.attr("reserved_length").cast<int64_t>();
    kv_storage_config.slot_byte_size =
        kv_storage_config_obj.attr("slot_byte_size").cast<int64_t>();
    kv_storage_config.storage_byte_size =
        kv_storage_config_obj.attr("storage_byte_size").cast<int64_t>();
    return kv_storage_config;
};

GPU_Buffer_Config parse_gpu_buffer_config(const py::object& engine_config) {
    GPU_Buffer_Config gpu_buffer_config;
    const py::object& gpu_buffer_config_obj =
        engine_config.attr("GPU_Buffer_Config");
    gpu_buffer_config.num_prefill_module_buffer =
        gpu_buffer_config_obj.attr("num_prefill_module_buffer")
            .cast<std::unordered_map<std::string, int64_t>>();
    gpu_buffer_config.num_decoding_module_buffer =
        gpu_buffer_config_obj.attr("num_decoding_module_buffer")
            .cast<std::unordered_map<std::string, int64_t>>();
    gpu_buffer_config.num_k_buffer =
        gpu_buffer_config_obj.attr("num_k_buffer").cast<int64_t>();
    gpu_buffer_config.num_v_buffer =
        gpu_buffer_config_obj.attr("num_v_buffer").cast<int64_t>();

    gpu_buffer_config.kv_buffer_num_tokens =
        gpu_buffer_config_obj.attr("kv_buffer_num_tokens").cast<int64_t>();
    py::dict module_shapes_py =
        gpu_buffer_config_obj.attr("module_shapes").cast<py::dict>();
    for (auto item : module_shapes_py) {
        std::string module_type = item.first.cast<std::string>();
        py::dict module_shape_dict = item.second.cast<py::dict>();
        std::unordered_map<std::string, std::vector<int64_t>> module_shape;
        for (auto item : module_shape_dict) {
            std::string module_name = item.first.cast<std::string>();
            std::vector<int64_t> shape =
                item.second.cast<std::vector<int64_t>>();
            module_shape[module_name] = shape;
        }
        gpu_buffer_config.module_shapes[module_type] = module_shape;
    }
    return gpu_buffer_config;
};

Module_Batching_Config parse_module_batching_config(
    const py::object& engine_config) {
    // std::cerr << "Parsing Module_Batching_Config" << std::endl;
    Module_Batching_Config module_batching_config;
    py::object module_batching_config_obj =
        engine_config.attr("Module_Batching_Config");
    // module_batching_config.prefill_micro_batch_size =
    // module_batching_config_obj.attr("prefill_micro_batch_size").cast<int64_t>();
    // module_batching_config.attn_decoding_micro_batch_size =
    // module_batching_config_obj.attr("attn_decoding_micro_batch_size").cast<int64_t>();
    // module_batching_config.expert_prefill_batch_size_upper_bound =
    // module_batching_config_obj.attr("expert_prefill_batch_size_upper_bound").cast<int64_t>();
    // module_batching_config.expert_decoding_batch_size_upper_bound =
    // module_batching_config_obj.attr("expert_decoding_batch_size_upper_bound").cast<int64_t>();
    // std::cerr << "Parsing Module_Batching_Config Done" << std::endl;
    return module_batching_config;
};

EngineConfig parse_engine_config(const py::object& engine_config) {
    // std::cerr << "Parsing EngineConfig" << std::endl;
    EngineConfig config;
    config.basic_config = parse_basic_config(engine_config);
    config.kv_storage_config = parse_kv_storage_config(engine_config);
    config.gpu_buffer_config = parse_gpu_buffer_config(engine_config);
    config.module_batching_config = parse_module_batching_config(engine_config);
    // std::cerr << "Parsing EngineConfig Done" << std::endl;
    return config;
};

ModelConfig parse_model_config(const py::object& model_config) {
    // std::cerr << "Parsing ModelConfig" << std::endl;
    ModelConfig config;
    config.model_type = model_config.attr("model_type").cast<std::string>();
    config.num_hidden_layers =
        model_config.attr("num_hidden_layers").cast<int64_t>();
    config.num_local_experts =
        model_config.attr("num_local_experts").cast<int64_t>();
    config.num_attention_heads =
        model_config.attr("num_attention_heads").cast<int64_t>();
    config.num_key_value_heads =
        model_config.attr("num_key_value_heads").cast<int64_t>();
    // config.hidden_size = model_config.attr("hidden_size").cast<int64_t>();
    // config.intermediate_size =
    // model_config.attr("intermediate_size").cast<int64_t>();
    config.head_dim = model_config.attr("head_dim").cast<int64_t>();
    if ((config.model_type == "deepseek_v2") ||
        (config.model_type == "deepseek_v3")) {
        config.compressed_kv_dim =
            model_config.attr("compressed_kv_dim").cast<int64_t>();
    }
    // std::cerr << "Parsing ModelConfig Done" << std::endl;
    return config;
};

std::shared_ptr<spdlog::logger> init_logger(const std::string& log_level,
                                            const std::string& logger_name) {
    auto logger = spdlog::stdout_color_mt(logger_name);

    // Set colors for all five standard levels
    auto console_sink = dynamic_cast<spdlog::sinks::stdout_color_sink_mt*>(
        logger->sinks()[0].get());
    if (console_sink) {
        console_sink->set_color(spdlog::level::trace, console_sink->white);
        console_sink->set_color(spdlog::level::debug, console_sink->cyan);
        console_sink->set_color(spdlog::level::info, console_sink->green);
        console_sink->set_color(spdlog::level::warn, console_sink->yellow);
        console_sink->set_color(spdlog::level::err, console_sink->red);
    }

    // Set the log level
    logger->set_level(spdlog::level::from_str(log_level));

    // Set the pattern based on the log level
    if (logger->level() <= spdlog::level::debug) {
        logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%n] %v");
    } else {
        logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    }

    // Make this logger the default logger
    // spdlog::set_default_logger(logger);

    // Optional: load log levels from the environment
    // spdlog::cfg::load_env_levels();
    logger->flush_on(spdlog::level::trace);
    return std::shared_ptr<spdlog::logger>(logger);
};
