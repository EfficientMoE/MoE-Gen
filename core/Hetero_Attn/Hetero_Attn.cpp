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

#include <condition_variable>
#include <cuda.h>
#include <cuda_runtime.h>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <queue>
#include <spdlog/spdlog.h>
#include <string>
#include <thread>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

#include "../DtoH_Engine/DtoH_Engine.h"
#include "../GPU_Weight_Buffer/GPU_Weight_Buffer.h"
#include "../HtoD_Engine/HtoD_Engine.h"
#include "../KV_Storage/KV_Storage.h"
#include "../Weights_Storage/Weights_Storage.h"
#include "../data_structures.h"
#include "../utils.h"
#include "CPU_Kernels/cpu_kernels.h"
#include "Hetero_Attn.h"

Hetero_Attn::Hetero_Attn(const EngineConfig& engine_config,
                         const ModelConfig& model_config,
                         KV_Storage& kv_storage, GPU_KV_Buffer& gpu_kv_buffer,
                         HtoD_Engine& h2d_engine, DtoH_Engine& d2h_engine)
    : engine_config_(engine_config),
      model_config_(model_config),
      kv_storage_(kv_storage),
      h2d_engine_(h2d_engine),
      d2h_engine_(d2h_engine),
      gpu_kv_buffer_(gpu_kv_buffer) {
    this->logger_ = init_logger(
        this->engine_config_.basic_config.log_level,
        "Hetero_Attn" +
            std::to_string(this->engine_config_.basic_config.device));
    this->attn_mode_ = this->engine_config_.basic_config.attn_mode;
};

torch::Tensor Hetero_Attn::attn(
    py::object& PyTorch_attn_module, int64_t layer_idx,
    torch::Tensor& hidden_states, torch::Tensor& attention_mask,
    torch::Tensor& position_ids,
    std::vector<std::vector<int64_t>> cur_batching_plan) {
    // Step 1: Checkings.
    if (!Py_IsInitialized()) {
        PyErr_Print();
        throw std::runtime_error("Python interpreter is not running!");
    };

    // Step 2: call
    switch (this->attn_mode_) {
        case 0:
            return this->_attn_mode_0(PyTorch_attn_module, layer_idx,
                                      hidden_states, attention_mask,
                                      position_ids, cur_batching_plan[0]);
        case 1:
            return this->_attn_mode_1(PyTorch_attn_module, layer_idx,
                                      hidden_states, attention_mask,
                                      position_ids, cur_batching_plan);
        case 2:
            return this->_attn_mode_2(PyTorch_attn_module, layer_idx,
                                      hidden_states, attention_mask,
                                      position_ids, cur_batching_plan);
        default:
            throw std::runtime_error("Unsupported attn_mode_");
    };
};

std::string get_tensor_shape(const torch::Tensor& t) {
    std::stringstream ss;
    ss << "[";
    for (int i = 0; i < t.dim(); ++i) {
        ss << t.size(i);
        if (i < t.dim() - 1) ss << ", ";
    }
    ss << "]";
    return ss.str();
}

torch::Tensor Hetero_Attn::_attn_mode_0(py::object& PyTorch_attn_module,
                                        int64_t layer_idx,
                                        torch::Tensor& hidden_states,
                                        torch::Tensor& attention_mask,
                                        torch::Tensor& position_ids,
                                        std::vector<int64_t> batch) {
    /*
      Single thread.
      - Call python module's QKV_Proj
      - Call full attn_mechanism kernel. attn_QKT_softmax_CPU + attn_AV
      - Call python module's O_Proj
    */
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));

    this->logger_->debug("MS 1");

    auto kv_seq_len = attention_mask.size(-1);
    auto qkv_result =
        PyTorch_attn_module
            .attr("QKV_Proj")(hidden_states, position_ids, kv_seq_len)
            .cast<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>();
    CUDA_CHECK(cudaStreamSynchronize(0));
    
    this->logger_->debug("QKV_Proj computed.");
    auto [query_states, key_states, value_states] = qkv_result;
    this->logger_->debug("QKV decomposed.");
    // try{
    //     attention_mask = this->d2h_engine_.tensor_on_demand_copy(attention_mask);
    //     this->logger_->debug("attn mask copied to host.");
    // } catch (const std::exception& e) {
    //     std::cerr << e.what() << '\n';
    //     throw;
    // } catch (...) {
    //     std::cerr << "Unknown error occurred." << std::endl;
    //     throw;
    // }
    attention_mask = attention_mask.to(torch::kCPU);

    this->logger_->debug("Attention mask shape: {}",
                         get_tensor_shape(attention_mask));
    // query_states = this->d2h_engine_.tensor_on_demand_copy(query_states);
    query_states = query_states.to(torch::kCPU);
    this->kv_storage_.update(layer_idx, batch, key_states, value_states);
    this->logger_->debug("K and V updated to the storage.");

    auto k_ptrs = this->kv_storage_.get_k_ptrs(layer_idx, batch);
    auto v_ptrs = this->kv_storage_.get_v_ptrs(layer_idx, batch);

    this->logger_->debug("K and V pointers size: {}", k_ptrs.size());
    this->logger_->debug("num threads: {}",
                         this->engine_config_.basic_config.num_threads);
    this->logger_->debug("query_states shape: {}",
                         get_tensor_shape(query_states));
    torch::Tensor attn_output;
    try{
        auto kernel_start = std::chrono::high_resolution_clock::now();
        attn_output = grouped_query_attention_cpu_avx2(
            query_states, k_ptrs, v_ptrs, attention_mask, attention_mask.size(-1),
            this->model_config_.num_attention_heads /
                this->model_config_.num_key_value_heads,
            this->model_config_.num_attention_heads,
            this->model_config_.num_key_value_heads, this->model_config_.head_dim,
            this->engine_config_.basic_config.num_threads);
        auto kernel_end = std::chrono::high_resolution_clock::now();
        auto kernel_duration =
            std::chrono::duration_cast<std::chrono::microseconds>(kernel_end -
                                                                kernel_start);
        this->logger_->debug("CPU Attn kernel duration: {} microseconds.",
                            kernel_duration.count());

        this->logger_->debug("CPU Attn output computed.");
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        throw;
    } catch (...) {
        std::cerr << "Unknown error occurred." << std::endl;
        throw;
    }
    attn_output = this->h2d_engine_.tensor_on_demand_copy(attn_output);

    auto final_output =
        PyTorch_attn_module.attr("O_Proj")(attn_output).cast<torch::Tensor>();
    CUDA_CHECK(cudaStreamSynchronize(0));
    // CUDA_CHECK(cudaDeviceSynchronize());

    return final_output;
};

torch::Tensor Hetero_Attn::_attn_mode_1(
    py::object& PyTorch_attn_module, int64_t layer_idx,
    torch::Tensor& hidden_states, torch::Tensor& attention_mask,
    torch::Tensor& position_ids,
    std::vector<std::vector<int64_t>> micro_batches) {
    /*
            FULL GPU MODE.
    */
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    // py::gil_scoped_acquire acquire;

    torch::Tensor final_output = torch::zeros_like(hidden_states);
    if (this->model_config_.model_type.find("deepseek") == std::string::npos) {
        for (int64_t micro_batch_idx = 0;
             micro_batch_idx < micro_batches.size(); micro_batch_idx++) {
            auto cur_batch = micro_batches[micro_batch_idx];
            auto cur_batch_size = cur_batch.size();

            int64_t bsz = cur_batch_size;
            int64_t kv_seq_len = attention_mask.size(-1) - 1;
            std::vector<int64_t> tensor_shape = {
                bsz, kv_seq_len, this->model_config_.num_key_value_heads,
                this->model_config_.head_dim};

            auto cur_k = this->gpu_kv_buffer_.get_k(layer_idx, micro_batch_idx,
                                                    tensor_shape);
            auto cur_v = this->gpu_kv_buffer_.get_v(layer_idx, micro_batch_idx,
                                                    tensor_shape);

            int64_t cur_batch_start_idx = 0;
            for (int64_t i = 0; i < micro_batch_idx; i++) {
                cur_batch_start_idx += micro_batches[i].size();
            };
            auto cur_hidden_states = hidden_states.index(
                {torch::indexing::Slice(cur_batch_start_idx,
                                        cur_batch_start_idx + cur_batch_size)});
            // CUDA_CHECK(cudaStreamSynchronize(0));
            // CUDA_CHECK(cudaDeviceSynchronize());
            auto module_output =
                PyTorch_attn_module
                    .attr("decoding_attn")(
                        cur_hidden_states, cur_k, cur_v,
                        attention_mask.index({torch::indexing::Slice(
                            cur_batch_start_idx,
                            cur_batch_start_idx + cur_batch_size)}),
                        position_ids.index({torch::indexing::Slice(
                            cur_batch_start_idx,
                            cur_batch_start_idx + cur_batch_size)}))
                    .cast<std::tuple<torch::Tensor, torch::Tensor,
                                     torch::Tensor>>();
            CUDA_CHECK(cudaStreamSynchronize(0));

            auto [attn_result, new_k, new_v] = module_output;
            this->gpu_kv_buffer_.releaseBuffer(layer_idx, micro_batch_idx);
            this->kv_storage_.update(layer_idx, cur_batch, new_k,
                                     new_v);  // Todo.
            // CUDA_CHECK(cudaStreamSynchronize(0));
            // CUDA_CHECK(cudaDeviceSynchronize());
            // result.push_back(attn_result);
            final_output.index_put_(
                {torch::indexing::Slice(cur_batch_start_idx,
                                        cur_batch_start_idx + cur_batch_size)},
                attn_result);
            // py::gil_scoped_release release;
        }
    } else {
        for (int64_t micro_batch_idx = 0;
             micro_batch_idx < micro_batches.size(); micro_batch_idx++) {
            auto cur_batch = micro_batches[micro_batch_idx];
            auto cur_batch_size = cur_batch.size();

            int64_t bsz = cur_batch_size;
            int64_t kv_seq_len = attention_mask.size(-1) - 1;
            std::vector<int64_t> tensor_shape = {
                bsz, kv_seq_len, this->model_config_.compressed_kv_dim};

            auto cur_k = this->gpu_kv_buffer_.get_k(layer_idx, micro_batch_idx,
                                                    tensor_shape);
            // auto cur_v = this->gpu_kv_buffer_.get_v(layer_idx,
            // micro_batch_idx, tensor_shape);

            int64_t cur_batch_start_idx = 0;
            for (int64_t i = 0; i < micro_batch_idx; i++) {
                cur_batch_start_idx += micro_batches[i].size();
            };
            auto cur_hidden_states = hidden_states.index(
                {torch::indexing::Slice(cur_batch_start_idx,
                                        cur_batch_start_idx + cur_batch_size)});
            // CUDA_CHECK(cudaStreamSynchronize(0));
            // CUDA_CHECK(cudaDeviceSynchronize());
            torch::Tensor cur_v = torch::empty(
                {0}, torch::TensorOptions()
                         .dtype(this->engine_config_.basic_config.dtype_torch)
                         .device(torch::kCUDA,
                                 this->engine_config_.basic_config.device)
                         .requires_grad(false)
                         .memory_format(torch::MemoryFormat::Contiguous));
            std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
                module_output;
            {
                py::gil_scoped_acquire acquire;
                module_output =
                    PyTorch_attn_module
                        .attr("decoding_attn")(
                            cur_hidden_states, cur_k, cur_v,
                            attention_mask.index({torch::indexing::Slice(
                                cur_batch_start_idx,
                                cur_batch_start_idx + cur_batch_size)}),
                            position_ids.index({torch::indexing::Slice(
                                cur_batch_start_idx,
                                cur_batch_start_idx + cur_batch_size)}))
                        .cast<std::tuple<torch::Tensor, torch::Tensor,
                                         torch::Tensor>>();
                py::gil_scoped_release release;
            }
            CUDA_CHECK(cudaStreamSynchronize(0));

            auto [attn_result, new_k, new_v] = module_output;
            this->gpu_kv_buffer_.releaseBuffer(layer_idx, micro_batch_idx);
            this->kv_storage_.update(layer_idx, cur_batch, new_k,
                                     new_v);  // Todo.
            // CUDA_CHECK(cudaStreamSynchronize(0));
            // CUDA_CHECK(cudaDeviceSynchronize());
            // result.push_back(attn_result);
            final_output.index_put_(
                {torch::indexing::Slice(cur_batch_start_idx,
                                        cur_batch_start_idx + cur_batch_size)},
                attn_result);
            // py::gil_scoped_release release;
        }
    }
    // auto final_output = torch::cat(result, 0);
    return final_output;
};

torch::Tensor Hetero_Attn::_attn_mode_2(
    py::object& PyTorch_attn_module, int64_t layer_idx,
    torch::Tensor& hidden_states, torch::Tensor& attention_mask,
    torch::Tensor& position_ids,
    std::vector<std::vector<int64_t>> micro_batches) {
    /*
            - Call python module's QKV_Proj
            - CPU GPU attn parallel.
            - Call python module's O_Proj
    */
    // if (this->model_config_.model_type.find("deepseek") != std::string::npos) {
    //     return this->_attn_mode_2_deepseekv2(PyTorch_attn_module, layer_idx,
    //                                          hidden_states, attention_mask,
    //                                          position_ids, micro_batches);
    // }

    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    auto CPU_Batch = micro_batches[0];
    std::vector<std::vector<int64_t>> GPU_Batch;
    for (int64_t i = 1; i < micro_batches.size(); i++) {
        GPU_Batch.push_back(micro_batches[i]);
    }
    auto kv_seq_len = attention_mask.size(-1);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> qkv_result;
    // {
    // 	py::gil_scoped_acquire acquire;
    qkv_result =
        PyTorch_attn_module
            .attr("QKV_Proj")(hidden_states, position_ids, kv_seq_len)
            .cast<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>();
    // 	py::gil_scoped_release release;
    // }
    // CUDA_CHECK(cudaStreamSynchronize(0));
    // CUDA_CHECK(cudaDeviceSynchronize());
    auto [query_states, key_states, value_states] = qkv_result;

    int64_t CPU_batch_start_idx = 0;
    int64_t CPU_batch_end_idx = CPU_Batch.size();
    int64_t GPU_batch_start_idx = CPU_Batch.size();
    int64_t GPU_batch_end_idx = query_states.size(0);

    auto CPU_query_states = query_states
                                .index({torch::indexing::Slice(
                                    CPU_batch_start_idx, CPU_batch_end_idx)})
                                .to(torch::kCPU);
    auto CPU_key_states = key_states.index(
        {torch::indexing::Slice(CPU_batch_start_idx, CPU_batch_end_idx)});
    auto CPU_value_states = value_states.index(
        {torch::indexing::Slice(CPU_batch_start_idx, CPU_batch_end_idx)});
    auto CPU_attention_mask = attention_mask
                                  .index({torch::indexing::Slice(
                                      CPU_batch_start_idx, CPU_batch_end_idx)})
                                  .to(torch::kCPU);

    std::future<torch::Tensor> CPU_result_future;
    CPU_result_future = this->CPU_attn_mechanism(
        layer_idx, CPU_query_states, CPU_key_states, CPU_value_states,
        CPU_attention_mask, CPU_Batch);

    auto GPU_query_states = query_states.index(
        {torch::indexing::Slice(GPU_batch_start_idx, GPU_batch_end_idx)});
    auto GPU_key_states = key_states.index(
        {torch::indexing::Slice(GPU_batch_start_idx, GPU_batch_end_idx)});
    auto GPU_value_states = value_states.index(
        {torch::indexing::Slice(GPU_batch_start_idx, GPU_batch_end_idx)});
    auto GPU_attention_mask = attention_mask.index(
        {torch::indexing::Slice(GPU_batch_start_idx, GPU_batch_end_idx)});
    GPU_attention_mask =
        this->h2d_engine_.tensor_on_demand_copy(GPU_attention_mask);
    // CUDA_CHECK(cudaStreamSynchronize(0));
    // CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<torch::Tensor> GPU_result;
    this->logger_->debug("GPU Batch size: {}", GPU_Batch.size());
    this->logger_->debug("GPU Computation Start.");
    auto GPU_start = std::chrono::high_resolution_clock::now();
    for (int64_t GPU_micro_batch_idx = 0;
         GPU_micro_batch_idx < GPU_Batch.size(); GPU_micro_batch_idx++) {
        auto cur_batch = GPU_Batch[GPU_micro_batch_idx];
        auto cur_batch_size = cur_batch.size();
        int64_t bsz = cur_batch_size;
        int64_t kv_seq_len = attention_mask.size(-1) - 1;
        std::vector<int64_t> tensor_shape = {
            bsz, kv_seq_len, this->model_config_.num_key_value_heads,
            this->model_config_.head_dim};
        auto cur_k = this->gpu_kv_buffer_.get_k(layer_idx, GPU_micro_batch_idx,
                                                tensor_shape);
        auto cur_v = this->gpu_kv_buffer_.get_v(layer_idx, GPU_micro_batch_idx,
                                                tensor_shape);
        this->logger_->debug("cur batch size: {}", cur_batch_size);
        // CUDA_CHECK(cudaStreamSynchronize(0));
        int64_t cur_batch_start_idx = 0;
        for (int64_t i = 0; i < GPU_micro_batch_idx; i++) {
            cur_batch_start_idx += GPU_Batch[i].size();
        };
        auto cur_query_states = GPU_query_states.index({torch::indexing::Slice(
            cur_batch_start_idx, cur_batch_start_idx + cur_batch_size)});
        auto new_k = GPU_key_states.index({torch::indexing::Slice(
            cur_batch_start_idx, cur_batch_start_idx + cur_batch_size)});
        auto new_v = GPU_value_states.index({torch::indexing::Slice(
            cur_batch_start_idx, cur_batch_start_idx + cur_batch_size)});

        cur_k = torch::cat({cur_k, new_k}, 2);
        cur_v = torch::cat({cur_v, new_v}, 2);

        torch::Tensor attn_output;
        this->logger_->debug(
            "starting attn computation for GPU micro batch: {}",
            GPU_micro_batch_idx);
        auto attn_weights =
            PyTorch_attn_module
                .attr("attn_QKT_softmax")(
                    cur_query_states, cur_k,
                    GPU_attention_mask.index({torch::indexing::Slice(
                        cur_batch_start_idx,
                        cur_batch_start_idx + cur_batch_size)}))
                .cast<torch::Tensor>();
        this->logger_->debug("attn_weights shape: {}",
                             get_tensor_shape(attn_weights));
        attn_output = PyTorch_attn_module.attr("attn_AV")(attn_weights, cur_v)
                          .cast<torch::Tensor>();

        // CUDA_CHECK(cudaStreamSynchronize(0));
        // CUDA_CHECK(cudaDeviceSynchronize());

        this->gpu_kv_buffer_.releaseBuffer(layer_idx, GPU_micro_batch_idx);
        this->kv_storage_.update(layer_idx, cur_batch, new_k, new_v);
        GPU_result.push_back(attn_output);
    };
    auto GPU_end = std::chrono::high_resolution_clock::now();
    auto GPU_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        GPU_end - GPU_start);
    this->logger_->debug("GPU Computation Duration: {} ms",
                         GPU_duration.count());
    // CUDA_CHECK(cudaStreamSynchronize(0));
    auto cat_start = std::chrono::high_resolution_clock::now();
    auto GPU_result_tensor = torch::cat(GPU_result, 0);
    auto cat_end = std::chrono::high_resolution_clock::now();
    auto cat_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        cat_end - cat_start);
    this->logger_->debug("GPU Result cat duration: {} ms",
                         cat_duration.count());
    this->logger_->debug("GPU Result shape: {}, start waiting for CPU result.",
                         get_tensor_shape(GPU_result_tensor));
    auto start = std::chrono::high_resolution_clock::now();
    auto CPU_result = CPU_result_future.get();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    this->logger_->debug("Wait for CPU result: {} ms", duration.count());

    auto attn_output = torch::cat({CPU_result, GPU_result_tensor}, 0);
    // CUDA_CHECK(cudaStreamSynchronize(0));
    // CUDA_CHECK(cudaDeviceSynchronize());
    auto final_output =
        PyTorch_attn_module.attr("O_Proj")(attn_output).cast<torch::Tensor>();
    // CUDA_CHECK(cudaStreamSynchronize(0));
    // CUDA_CHECK(cudaDeviceSynchronize());

    return final_output;
};

// torch::Tensor Hetero_Attn::_attn_mode_2_deepseekv2(
//     py::object& PyTorch_attn_module, int64_t layer_idx,
//     torch::Tensor& hidden_states, torch::Tensor& attention_mask,
//     torch::Tensor& position_ids,
//     std::vector<std::vector<int64_t>> micro_batches) {
//     try {
//         CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
//         auto CPU_Batch = micro_batches[0];
//         std::vector<std::vector<int64_t>> GPU_Batch;
//         for (int64_t i = 1; i < micro_batches.size(); i++) {
//             GPU_Batch.push_back(micro_batches[i]);
//         }
//         std::vector<int64_t> full_batch = CPU_Batch;
//         for (int64_t i = 0; i < GPU_Batch.size(); i++) {
//             full_batch.insert(full_batch.end(), GPU_Batch[i].begin(),
//                               GPU_Batch[i].end());
//         }
//         int64_t kv_seq_len = attention_mask.size(-1) - 1;
//         std::vector<int64_t> past_kv_shape = {
//             full_batch.size(), kv_seq_len,
//             this->model_config_.compressed_kv_dim};
//         auto past_compressed_kv = this->gpu_kv_buffer_.get_k(
//             layer_idx, 0, past_kv_shape);  // Copy call kv in one go.
//         auto pre_attn_result =
//             PyTorch_attn_module
//                 .attr("Pre_Attn")(hidden_states, past_compressed_kv,
//                                   attention_mask, position_ids)
//                 .cast<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
//                                  torch::Tensor>>();
//         // CUDA_CHECK(cudaStreamSynchronize(0));
//         auto [new_compressed_kv, query_states, key_states, value_states] =
//             pre_attn_result;
//         this->gpu_kv_buffer_.releaseBuffer(layer_idx, 0);

//         int64_t CPU_batch_start_idx = 0;
//         int64_t CPU_batch_end_idx = CPU_Batch.size();
//         int64_t GPU_batch_start_idx = CPU_Batch.size();
//         int64_t GPU_batch_end_idx = query_states.size(0);

//         auto CPU_query_states = query_states.index(
//             {torch::indexing::Slice(CPU_batch_start_idx, CPU_batch_end_idx)});
//         auto CPU_key_states = key_states.index(
//             {torch::indexing::Slice(CPU_batch_start_idx, CPU_batch_end_idx)});
//         auto CPU_value_states = value_states.index(
//             {torch::indexing::Slice(CPU_batch_start_idx, CPU_batch_end_idx)});
//         auto CPU_attention_mask = attention_mask.index(
//             {torch::indexing::Slice(CPU_batch_start_idx, CPU_batch_end_idx)});

//         std::thread update_new_compressed_kv_thread(
//             [this, layer_idx, full_batch, new_compressed_kv]() {
//                 this->kv_storage_.update(layer_idx, full_batch,
//                                          new_compressed_kv, new_compressed_kv);
//             });

//         std::future<torch::Tensor> CPU_result_future;
//         CPU_result_future = this->CPU_attn_mechanism_deepseekv2(
//             layer_idx, CPU_query_states, CPU_key_states, CPU_value_states,
//             CPU_attention_mask, CPU_Batch);

//         auto GPU_query_states = query_states.index(
//             {torch::indexing::Slice(GPU_batch_start_idx, GPU_batch_end_idx)});
//         auto GPU_key_states = key_states.index(
//             {torch::indexing::Slice(GPU_batch_start_idx, GPU_batch_end_idx)});
//         auto GPU_value_states = value_states.index(
//             {torch::indexing::Slice(GPU_batch_start_idx, GPU_batch_end_idx)});
//         auto GPU_attention_mask =
//             attention_mask
//                 .index({torch::indexing::Slice(GPU_batch_start_idx,
//                                                GPU_batch_end_idx)})
//                 .to(this->engine_config_.basic_config.device_torch);

//         this->logger_->debug("Starting GPU computation.");
//         std::vector<torch::Tensor> GPU_result;
//         for (int64_t GPU_micro_batch_idx = 0;
//              GPU_micro_batch_idx < GPU_Batch.size(); GPU_micro_batch_idx++) {
//             auto cur_batch = GPU_Batch[GPU_micro_batch_idx];
//             auto cur_batch_size = cur_batch.size();
//             int64_t bsz = cur_batch_size;
//             int64_t cur_batch_start_idx = 0;
//             for (int64_t i = 0; i < GPU_micro_batch_idx; i++) {
//                 cur_batch_start_idx += GPU_Batch[i].size();
//             };
//             this->logger_->debug("GPU micro batch: {}", GPU_micro_batch_idx);
//             auto attn_output =
//                 PyTorch_attn_module
//                     .attr("Attn_Mechanism")(
//                         GPU_query_states.index({torch::indexing::Slice(
//                             cur_batch_start_idx,
//                             cur_batch_start_idx + cur_batch_size)}),
//                         GPU_key_states.index({torch::indexing::Slice(
//                             cur_batch_start_idx,
//                             cur_batch_start_idx + cur_batch_size)}),
//                         GPU_value_states.index({torch::indexing::Slice(
//                             cur_batch_start_idx,
//                             cur_batch_start_idx + cur_batch_size)}),
//                         GPU_attention_mask.index({torch::indexing::Slice(
//                             cur_batch_start_idx,
//                             cur_batch_start_idx + cur_batch_size)}))
//                     .cast<torch::Tensor>();
//             this->logger_->debug("Attn output shape: {}",
//                                  get_tensor_shape(attn_output));
//             GPU_result.push_back(attn_output);
//         };
//         auto GPU_result_tensor = torch::cat(GPU_result, 0);
//         auto CPU_result = CPU_result_future.get();
//         auto attn_output = torch::cat({CPU_result, GPU_result_tensor}, 0);
//         update_new_compressed_kv_thread.join();
//         auto final_output = PyTorch_attn_module.attr("Post_Attn")(attn_output)
//                                 .cast<torch::Tensor>();
//         // CUDA_CHECK(cudaStreamSynchronize(0));
//         return final_output;
//     } catch (const std::exception& e) {
//         std::cerr << e.what() << '\n';
//         throw;
//     } catch (...) {
//         std::cerr << "Unknown error occurred." << std::endl;
//         throw;
//     }
// }

std::future<torch::Tensor> Hetero_Attn::CPU_attn_mechanism(
    int64_t layer_idx, torch::Tensor query_states, torch::Tensor& key_states,
    torch::Tensor& value_states, torch::Tensor& attention_mask,
    std::vector<int64_t> batch) {
    auto promise = std::make_shared<std::promise<torch::Tensor>>();
    std::future<torch::Tensor> future = promise->get_future();
    std::thread([this, promise, query_states, key_states, value_states,
                 attention_mask, layer_idx, batch]() {
        try {
            // auto start = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
            this->kv_storage_.update(layer_idx, batch, key_states,
                                     value_states);
            this->logger_->debug("num threads: {}",
                                 this->engine_config_.basic_config.num_threads);
            auto k_ptrs = this->kv_storage_.get_k_ptrs(layer_idx, batch);
            auto v_ptrs = this->kv_storage_.get_v_ptrs(layer_idx, batch);
            auto start = std::chrono::high_resolution_clock::now();
            this->logger_->debug("Calling CPU kernel.");
            auto attn_output = grouped_query_attention_cpu_avx2(
                query_states, k_ptrs, v_ptrs, attention_mask,
                attention_mask.size(-1),
                this->model_config_.num_attention_heads /
                    this->model_config_.num_key_value_heads,
                this->model_config_.num_attention_heads,
                this->model_config_.num_key_value_heads,
                this->model_config_.head_dim,
                this->engine_config_.basic_config.num_threads);
            // attn_output =
            // this->h2d_engine_.tensor_on_demand_copy(attn_output);
            attn_output =
                attn_output.to(this->engine_config_.basic_config.device_torch);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                      start);
            this->logger_->debug("CPU Attn kernel duration: {} ms",
                                 duration.count());
            auto set_start = std::chrono::high_resolution_clock::now();
            promise->set_value(attn_output);
            auto set_end = std::chrono::high_resolution_clock::now();
            auto set_duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    set_end - set_start);
            this->logger_->debug("CPU Attn set value duration: {} ms",
                                 set_duration.count());
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    }).detach();

    return future;
};

// std::future<torch::Tensor> Hetero_Attn::CPU_attn_mechanism_deepseekv2(
//     int64_t layer_idx, torch::Tensor query_states, torch::Tensor& key_states,
//     torch::Tensor& value_states, torch::Tensor& attention_mask,
//     std::vector<int64_t> batch) {
//     CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
//     auto promise = std::make_shared<std::promise<torch::Tensor>>();
//     std::future<torch::Tensor> future = promise->get_future();
//     std::thread([this, promise, query_states, key_states, value_states,
//                  attention_mask, layer_idx, batch]() {
//         try {
//             auto start = std::chrono::high_resolution_clock::now();
//             auto cpu_query_states = query_states.to(torch::kCPU).contiguous();
//             auto cpu_key_states = key_states.to(torch::kCPU).contiguous();
//             auto cpu_value_states = value_states.to(torch::kCPU).contiguous();
//             auto cpu_attention_mask =
//                 attention_mask.to(torch::kCPU).contiguous();
//             std::vector<c10::BFloat16*> k_ptrs;
//             std::vector<c10::BFloat16*> v_ptrs;
//             for (int64_t i = 0; i < key_states.size(0); i++) {
//                 k_ptrs.push_back(key_states[i].data_ptr<c10::BFloat16>());
//                 v_ptrs.push_back(value_states[i].data_ptr<c10::BFloat16>());
//             }

//             auto attn_output = grouped_query_attention_cpu_avx2_deepseekv2(
//                 cpu_query_states, k_ptrs, v_ptrs, cpu_attention_mask,
//                 attention_mask.size(-1),
//                 this->model_config_.num_attention_heads /
//                     this->model_config_.num_key_value_heads,
//                 this->model_config_.num_attention_heads,
//                 this->model_config_.num_key_value_heads, 192, 128,
//                 this->engine_config_.basic_config.num_threads);
//             attn_output =
//                 attn_output.to(this->engine_config_.basic_config.device_torch);
//             auto end = std::chrono::high_resolution_clock::now();
//             auto duration =
//                 std::chrono::duration_cast<std::chrono::milliseconds>(end -
//                                                                       start);
//             this->logger_->debug("CPU Attn kernel duration: {} ms",
//                                  duration.count());
//             promise->set_value(attn_output);
//         } catch (...) {
//             promise->set_exception(std::current_exception());
//         }
//     }).detach();

//     return future;
// };
