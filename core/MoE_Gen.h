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

#pragma once
#include "spdlog/spdlog.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

#include "DtoH_Engine/DtoH_Engine.h"
#include "GPU_KV_Buffer/GPU_KV_Buffer.h"
#include "GPU_Weight_Buffer/GPU_Weight_Buffer.h"
#include "Hetero_Attn/Hetero_Attn.h"
#include "HtoD_Engine/HtoD_Engine.h"
#include "KV_Storage/KV_Storage.h"
#include "Weights_Storage/Weights_Storage.h"
#include "data_structures.h"
#include "threadsafe_queue.h"
#include "utils.h"

namespace py = pybind11;

class MoE_Gen {
   public:
    MoE_Gen(py::object engine_config, py::object model_config);

    ~MoE_Gen();

    void Init(std::string& shm_name, std::string& tensor_meta_shm_name,
              int64_t byte_size,
              std::unordered_map<std::string, std::vector<std::string>>&
                  weights_copy_tasks);
    void Terminate();

    /*
            Exposing APIs to PyTorch Side.
    */
    void prefill_complete_sync();
    // std::unordered_map<std::string, torch::Tensor> get_skeleton_state_dict();
    /* Holding */
    void set_batching_plan(std::vector<std::vector<int64_t>>
                               batching_plan);  // Called every round.

    void submit_to_KV_queue(std::vector<int64_t> micro_batch,
                            int64_t micro_batch_idx, int64_t layer_idx,
                            int64_t byte_size);

    /* KV */
    void kv_offload(int64_t layer_idx, std::vector<int64_t> query_idx,
                    torch::Tensor key_states, torch::Tensor value_states);

    /* Weights */
    // void add_weight_storage(
    // 	const std::string& module_key,
    // 	const std::string& tensor_key,
    // 	int64_t tensor_ptr,
    // 	const std::vector<int64_t>& tensor_shape,
    // 	int64_t byte_size
    // );

    std::unordered_map<std::string, torch::Tensor> get_weights(
        std::string module_key);
    void free_weights_buffer(const std::string& module_name);

    /* Attn */
    torch::Tensor attn(py::object torch_module, int64_t layer_idx,
                       torch::Tensor& hidden_states,
                       torch::Tensor& attention_mask,
                       torch::Tensor& position_ids,
                       std::vector<std::vector<int64_t>> cur_batching_plan);

    void clear_expert_buffer(int64_t layer_idx, int64_t expert_idx);

    void set_phase(std::string phase);

    void register_signal_handler();

    void clear_kv_storage();
    void clear_kv_buffer();
    void clear_kv_copy_queue();
    void clear_weight_copy_queue();
    void reset_weight_copy_queue();
    void reset_prefill_buffer();

   private:
    EngineConfig engine_config_;
    ModelConfig model_config_;
    std::shared_ptr<spdlog::logger> logger;

    std::string phase_;
    std::string shm_name_;
    std::string tensor_meta_shm_name_;
    KV_Storage kv_storage_;
    Weights_Storage weights_storage_;
    HtoD_Engine h2d_engine_;
    DtoH_Engine d2h_engine_;
    GPU_Weight_Buffer gpu_weight_buffer_;
    GPU_KV_Buffer gpu_kv_buffer_;
    py::object parameter_server_;

    Hetero_Attn hetero_attn_;
};
