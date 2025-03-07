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
#include <memory>
#include <string>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

#include "../KV_Storage/KV_Storage.h"
#include "../Weights_Storage/Weights_Storage.h"
#include "../data_structures.h"
#include "../threadsafe_queue.h"
#include "../utils.h"
#include "HtoD_Engine.h"

HtoD_Engine::HtoD_Engine(const EngineConfig& engine_config,
                         const ModelConfig& model_config,
                         Weights_Storage& weights_storage,
                         KV_Storage& kv_storage,
                         GPU_Weight_Buffer& gpu_weight_buffer,
                         GPU_KV_Buffer& gpu_kv_buffer)
    : engine_config_(engine_config),
      model_config_(model_config),
      weights_storage_(weights_storage),
      kv_storage_(kv_storage),
      gpu_weight_buffer_(gpu_weight_buffer),
      gpu_kv_buffer_(gpu_kv_buffer) {
    cudaSetDevice(this->engine_config_.basic_config.device);
    this->logger_ = init_logger(
        this->engine_config_.basic_config.log_level,
        "HtoD_Engine" +
            std::to_string(this->engine_config_.basic_config.device));
    CUDA_CHECK(
        cudaStreamCreateWithFlags(&this->HtoD_stream, cudaStreamNonBlocking));
};

void HtoD_Engine::Init(
    std::unordered_map<std::string, std::vector<std::string>>&
        weight_copy_tasks) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->terminate_flag_ = false;
    this->weight_copy_tasks_ = weight_copy_tasks;
    for (auto& item : weight_copy_tasks) {
        auto module_type = item.first;
        auto module_names = item.second;
        this->weights_copy_task_queue_.emplace(module_type,
                                               threadsafe_queue<std::string>());
        for (auto& module_name : module_names) {
            this->weights_copy_task_queue_[module_type].push(module_name);
        }
    }
    this->HtoD_worker_ = std::thread(&HtoD_Engine::HtoD_Worker, this);
};

void HtoD_Engine::Terminate() {
    this->terminate_flag_ = true;
    if (this->HtoD_worker_.joinable()) {
        this->HtoD_worker_.join();
    }
};

HtoD_Engine::~HtoD_Engine() { this->Terminate(); };

torch::Tensor HtoD_Engine::tensor_on_demand_copy(torch::Tensor src_tensor) {
    auto dst_tensor = torch::empty_like(src_tensor, torch::kCUDA).contiguous();
    // Create packaged task
    std::packaged_task<void()> task(
        [this, dst_ptr = dst_tensor.data_ptr(), src_ptr = src_tensor.data_ptr(),
         size = src_tensor.numel() * src_tensor.element_size()]() {
            this->blocking_copy_(dst_ptr, src_ptr, size);
        });

    // Get future before moving task
    std::future<void> completion_future = task.get_future();

    // Push task to queue
    this->on_demand_task_queue_.push(std::move(task));

    // Wait for completion
    completion_future.wait();

    return dst_tensor;
};

void HtoD_Engine::submit_to_KV_queue(std::vector<int64_t>& micro_batch,
                                     int64_t micro_batch_idx, int64_t layer_idx,
                                     int64_t byte_size) {
    this->kv_copy_task_queue_.push(
        std::make_tuple(micro_batch, micro_batch_idx, layer_idx, byte_size));
};

void HtoD_Engine::blocking_copy_(void* dst, void* src, int64_t size) {
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice,
                               this->HtoD_stream));
    // CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaStreamSynchronize(this->HtoD_stream));
};

void HtoD_Engine::HtoD_Worker() {
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    while (!terminate_flag_) {
        std::packaged_task<void()> task;
        while (on_demand_task_queue_.try_pop(task)) {
            task();
        }
        if (!this->kv_copy_task_queue_.empty()) {
            auto optional_buffer = this->gpu_kv_buffer_.acquireEmptyBuffer();
            if (optional_buffer.has_value()) {
                if (this->model_config_.model_type.find("deepseek") ==
                    std::string::npos) {
                    // this->logger_->debug("DeepSeek model detected. Copying
                    // both K to GPU.");
                    auto [dst_k_ptr, dst_v_ptr, buffer_idx] =
                        optional_buffer.value();
                    std::tuple<std::vector<int64_t>, int64_t, int64_t, int64_t>
                        task;
                    this->kv_copy_task_queue_.wait_and_pop(task);
                    auto& [cur_batch, micro_batch_idx, layer_idx, byte_size] =
                        task;
                    this->logger_->debug(
                        "copying micro_batch_idx: {}, layer_idx: {}, "
                        "byte_size: {}",
                        micro_batch_idx, layer_idx, byte_size);
                    auto host_k_ptrs =
                        this->kv_storage_.get_k_ptrs(layer_idx, cur_batch);
                    auto host_v_ptrs =
                        this->kv_storage_.get_v_ptrs(layer_idx, cur_batch);
                    int64_t k_offset = 0;
                    int64_t v_offset = 0;
                    int64_t k_byte_size = byte_size;
                    int64_t v_byte_size = byte_size;
                    for (int64_t i = 0; i < cur_batch.size(); i++) {
                        CUDA_CHECK(cudaMemcpyAsync(
                            dst_k_ptr + k_offset, host_k_ptrs[i], k_byte_size,
                            cudaMemcpyHostToDevice, this->HtoD_stream));
                        CUDA_CHECK(cudaMemcpyAsync(
                            dst_v_ptr + v_offset, host_v_ptrs[i], v_byte_size,
                            cudaMemcpyHostToDevice, this->HtoD_stream));
                        k_offset += k_byte_size;
                        v_offset += v_byte_size;
                    }
                    CUDA_CHECK(cudaStreamSynchronize(this->HtoD_stream));
                    this->gpu_kv_buffer_.kv_copy_complete(
                        layer_idx, micro_batch_idx, buffer_idx);
                    this->logger_->debug("Copied KV to buffer: {}", buffer_idx);
                } else {
                    auto [dst_k_ptr, dst_v_ptr, buffer_idx] =
                        optional_buffer.value();
                    std::tuple<std::vector<int64_t>, int64_t, int64_t, int64_t>
                        task;
                    this->kv_copy_task_queue_.wait_and_pop(task);
                    auto& [cur_batch, micro_batch_idx, layer_idx, byte_size] =
                        task;
                    this->logger_->debug(
                        "copying micro_batch_idx: {}, layer_idx: {}, "
                        "byte_size: {}",
                        micro_batch_idx, layer_idx, byte_size);
                    auto host_k_ptrs =
                        this->kv_storage_.get_k_ptrs(layer_idx, cur_batch);
                    // auto host_v_ptrs =
                    // this->kv_storage_.get_v_ptrs(layer_idx, cur_batch);
                    int64_t k_offset = 0;
                    // int64_t v_offset = 0;
                    int64_t k_byte_size = byte_size;
                    // int64_t v_byte_size = byte_size;
                    for (int64_t i = 0; i < cur_batch.size(); i++) {
                        CUDA_CHECK(cudaMemcpyAsync(
                            dst_k_ptr + k_offset, host_k_ptrs[i], k_byte_size,
                            cudaMemcpyHostToDevice, this->HtoD_stream));
                        k_offset += k_byte_size;
                    }
                    CUDA_CHECK(cudaStreamSynchronize(this->HtoD_stream));
                    this->gpu_kv_buffer_.kv_copy_complete(
                        layer_idx, micro_batch_idx, buffer_idx);
                    this->logger_->debug("Copied KV to buffer: {}", buffer_idx);
                }
            }
        };

        for (auto& module_type :
             this->engine_config_.basic_config.module_types) {
            auto optional_buffer =
                this->gpu_weight_buffer_.acquireEmptyBuffer(module_type);
            if (optional_buffer.has_value()) {
                this->logger_->debug("Acquired buffer for module type: {}",
                                     module_type);
                auto [buffer, buffer_idx] = optional_buffer.value();
                std::string module_name;
                while (this->weights_copy_task_queue_[module_type].try_pop(
                           module_name) == false) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
                auto dst = buffer.get();
                auto src = this->weights_storage_.get_module_weights_storage(
                    module_name);
                // For DeepSeek-R1 and V3, convert to bf16 before copying to
                // GPU.
                torch::Tensor tmp_src;
                void* src_ptr;
                int64_t src_byte_size;
                for (auto& [tensor_name, host_tensor_storage] : src) {
                    // this->logger_->debug("Copying tensor. Src: {}, Dst: {},
                    // Byte_size:
                    // {}", host_tensor_storage.data_ptr,
                    // dst[tensor_name].data_ptr(),
                    // host_tensor_storage.byte_size);
                    // if(this->model_config_.model_type
                    // == "deepseek_v3"){ 	tmp_src =
                    // torch::from_blob(host_tensor_storage.data_ptr,
                    // host_tensor_storage.tensor_shape, torch::kFloat8_e4m3fn);
                    // tmp_src = tmp_src.to(torch::kBFloat16); 	src_ptr =
                    // tmp_src.data_ptr(); 	src_byte_size = tmp_src.numel()
                    // * tmp_src.element_size();
                    // }
                    // else{
                    // 	src_ptr = host_tensor_storage.data_ptr;
                    // 	src_byte_size = host_tensor_storage.byte_size;
                    // }
                    src_ptr = host_tensor_storage.data_ptr;
                    src_byte_size = host_tensor_storage.byte_size;
                    // this->blocking_copy_(dst[tensor_name].data_ptr(),
                    // src_ptr,
                    //                      src_byte_size);
                    if (dst[tensor_name].defined() &&
                        dst[tensor_name].has_storage()) {
                        this->blocking_copy_(dst[tensor_name].data_ptr(),
                                             src_ptr, src_byte_size);
                    } else {
                        this->logger_->error(
                            "Tensor {} doesn't have valid storage",
                            tensor_name);
                        std::runtime_error("Tensor doesn't have valid storage");
                    }
                }
                this->logger_->debug("Copied module: {} to buffer: {}",
                                     module_name, buffer_idx);
                this->gpu_weight_buffer_.weights_copy_complete(
                    module_type, module_name, buffer_idx);
                /* PUSH THE TASK BACK */
                {
                    std::lock_guard<std::mutex> lock(this->mutex_);
                    this->weights_copy_task_queue_[module_type].push(
                        module_name);
                }
            }
        }
    }
};

void HtoD_Engine::clear_kv_copy_queue() {
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    try {
        while (!this->kv_copy_task_queue_.empty()) {
            std::tuple<std::vector<int64_t>, int64_t, int64_t, int64_t> task;
            this->kv_copy_task_queue_.wait_and_pop(task);
        }
    } catch (...) {
        this->logger_->error("Failed to clear kv_copy_queue.");
    }
    CUDA_CHECK(cudaStreamSynchronize(this->HtoD_stream));
};

void HtoD_Engine::clear_weight_copy_queue() {
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    std::lock_guard<std::mutex> lock(this->mutex_);
    for (auto& [module_type, queue] : this->weights_copy_task_queue_) {
        while (!queue.empty()) {
            queue.clear();
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(this->HtoD_stream));
}

void HtoD_Engine::reset_weight_copy_queue() {
    std::lock_guard<std::mutex> lock(this->mutex_);

    // re-init the weights_copy_task_queue_ according to the weight_copy_tasks
    auto new_weights_copy_task_queue =
        std::unordered_map<std::string, threadsafe_queue<std::string>>();
    for (auto& item : this->weight_copy_tasks_) {
        auto module_type = item.first;
        auto module_names = item.second;
        new_weights_copy_task_queue.emplace(module_type,
                                            threadsafe_queue<std::string>());
        for (auto& module_name : module_names) {
            new_weights_copy_task_queue[module_type].push(module_name);
        }
    }
    this->weights_copy_task_queue_ = new_weights_copy_task_queue;
};
