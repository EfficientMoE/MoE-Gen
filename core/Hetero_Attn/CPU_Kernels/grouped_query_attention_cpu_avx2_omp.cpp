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

#include <c10/util/BFloat16.h>
#include <chrono>
#include <cstdlib>
#include <immintrin.h>
#include <omp.h>
#include <stdexcept>
#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>

static inline __m256 bf16_to_fp32(__m128i bf16_vals) {
    __m256i tmp = _mm256_cvtepu16_epi32(bf16_vals);
    tmp = _mm256_slli_epi32(tmp, 16);
    return _mm256_castsi256_ps(tmp);
};

inline c10::BFloat16 dot_product(const c10::BFloat16* a, const c10::BFloat16* b,
                                 int64_t num_elements) {
    auto tmp = 0.0f;
    // Process 8 elements at a time
    for (int64_t i = 0; i < num_elements; i += 8) {
        // Direct load of 8 BF16 values (128 bits) into vector registers
        __m128i a_bf16 = _mm_loadu_si128((__m128i*)(&a[i]));
        __m128i b_bf16 = _mm_loadu_si128((__m128i*)(&b[i]));

        __m256 a_fp32 = bf16_to_fp32(a_bf16);
        __m256 b_fp32 = bf16_to_fp32(b_bf16);

        // Compute dot product with mask 0xf1 to sum all elements
        __m256 result = _mm256_dp_ps(a_fp32, b_fp32, 0xf1);

        // Unpack the result into an array of floats
        float unpacked[8];
        std::memcpy(unpacked, &result, sizeof(float) * 8);
        tmp += unpacked[0] + unpacked[4];
    }
    return c10::BFloat16(tmp);
};

inline void scalar_vector_multiply_add(const c10::BFloat16 scalar,
                                       const c10::BFloat16* vector, float* c,
                                       int64_t num_elements) {
    // Load bf16 scalar, convert to fp32 and broadcast to 8 lanes
    float scalar_float = float(scalar);  // Uses BFloat16's conversion operator
    __m256 scalar_fp32_vec = _mm256_set1_ps(scalar_float);

    // Process 8 elements at a time
    for (int64_t i = 0; i < num_elements; i += 8) {
        // Direct load of 8 BF16 values (128 bits) into vector registers
        __m128i b_bf16 = _mm_loadu_si128((__m128i*)(&vector[i]));

        // Convert BF16 to FP32
        __m256 b_fp32 = bf16_to_fp32(b_bf16);

        // Load existing values from c
        __m256 c_fp32 = _mm256_loadu_ps(c + i);

        // Multiply and add: c = scalar * b + c
        c_fp32 = _mm256_fmadd_ps(scalar_fp32_vec, b_fp32, c_fp32);

        // Store result back to memory
        _mm256_storeu_ps(c + i, c_fp32);
    }
};

torch::Tensor grouped_query_attention_cpu_avx2(
    const torch::Tensor& query_states,  // [bsz, num_attn_head, 1, head_dim]
    const std::vector<c10::BFloat16*>&
        key_states,  // bsz x [cache_length, num_kv_head, head_dim]
    const std::vector<c10::BFloat16*>&
        value_states,  // bsz x [cache_length, num_kv_head, head_dim]
    const torch::Tensor& attention_mask,  // [bsz, cache_length]
    int64_t cache_length, int64_t group_size, int64_t num_attention_heads,
    int64_t num_kv_head, int64_t head_dim, int64_t num_threads) {
    // std::cerr<<"attention_mask shape: "<<attention_mask.sizes()<<std::endl;
    // std::cerr<<"Cache length: "<<cache_length<<std::endl;
    // std::cerr<<"Group size: "<<group_size<<std::endl;
    // std::cerr<<"Num attention heads: "<<num_attention_heads<<std::endl;
    // std::cerr<<"Num kv head: "<<num_kv_head<<std::endl;
    // std::cerr<<"Head dim: "<<head_dim<<std::endl;

    try {
        int64_t bsz = query_states.size(0);
        if (bsz != key_states.size() || bsz != value_states.size()) {
            std::cout << "bsz: " << bsz
                      << ", key_states.size(): " << key_states.size()
                      << ", value_states.size(): " << value_states.size()
                      << std::endl;
            throw std::runtime_error("Batch size mismatch");
        }
        // torch::Tensor attn_output = torch::zeros({bsz, num_attention_heads,
        // 1, head_dim}, query_states.options());
        omp_set_num_threads(num_threads);

        // Step 1: Compute the attention weights(attention_scores).
        c10::BFloat16* attn_weights = static_cast<c10::BFloat16*>(
            std::malloc(bsz * num_attention_heads * 1 * cache_length *
                        sizeof(c10::BFloat16)));

        c10::BFloat16* attn_output = static_cast<c10::BFloat16*>(std::malloc(
            bsz * num_attention_heads * 1 * head_dim * sizeof(c10::BFloat16)));
        // memset
        memset(
            attn_output, 0,
            bsz * num_attention_heads * 1 * head_dim * sizeof(c10::BFloat16));

        // std::cerr<<"grouped_query_attention_cpu_avx2 start
        // parallel."<<std::endl;
        int chunk_size;
        if (bsz / num_threads / 10 < 1) {
            chunk_size = bsz / num_threads;
        } else {
            chunk_size = bsz / num_threads / 10;
        }
        // print log with cur time
        // std::cerr<<"grouped_query_attention_cpu_avx2 start
        // parallel."<<std::endl;
        auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel
        {
// int tid = omp_get_thread_num();
// cpu_set_t cpuset;
// CPU_ZERO(&cpuset);

// // Example strategy: bind thread 'tid' to CPU core 'tid'
// CPU_SET(tid, &cpuset);

// // Set affinity for the calling thread
// int ret = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
// if (ret != 0) {
// 	perror("sched_setaffinity");
// }
#pragma omp for schedule(static, 1)
            for (int64_t b = 0; b < bsz; ++b) {
                for (int64_t token = 0; token < cache_length; ++token) {
                    for (int64_t kv_head = 0; kv_head < num_kv_head;
                         ++kv_head) {
                        for (int64_t id_in_group = 0; id_in_group < group_size;
                             ++id_in_group) {
                            int64_t cur_attention_head =
                                kv_head * group_size + id_in_group;
                            auto dot_prod = dot_product(
                                static_cast<c10::BFloat16*>(
                                    query_states.data_ptr()) +
                                    b * num_attention_heads * 1 * head_dim +
                                    cur_attention_head * 1 * head_dim,
                                key_states[b] + token * num_kv_head * head_dim +
                                    kv_head * head_dim,
                                head_dim);
                            c10::BFloat16 scaled_dot_prod = c10::BFloat16(
                                static_cast<float>(dot_prod) /
                                std::sqrt(static_cast<float>(head_dim)));
                            c10::BFloat16 attention_weight = c10::BFloat16(
                                static_cast<float>(scaled_dot_prod) +
                                static_cast<float>(
                                    attention_mask.data_ptr<c10::BFloat16>()
                                        [b * cache_length + token]));
                            attn_weights[b * num_attention_heads * 1 *
                                             cache_length +
                                         cur_attention_head * 1 * cache_length +
                                         token] = attention_weight;
                        }
                    }
                }
                for (int64_t n = 0; n < num_attention_heads; ++n) {
                    float acc = 0.0f;
                    for (int64_t q = 0; q < cache_length; ++q) {
                        acc += std::exp(static_cast<float>(
                            attn_weights[b * num_attention_heads * 1 *
                                             cache_length +
                                         n * 1 * cache_length + q]));
                    }
                    for (int64_t q = 0; q < cache_length; ++q) {
                        attn_weights[b * num_attention_heads * 1 *
                                         cache_length +
                                     n * 1 * cache_length + q] =
                            c10::BFloat16(
                                std::exp(static_cast<float>(
                                    attn_weights[b * num_attention_heads * 1 *
                                                     cache_length +
                                                 n * 1 * cache_length + q])) /
                                acc);
                    }
                }
                // for(int64_t n = 0; n < num_attention_heads; ++n) {
                // 	// First find the maximum value
                // 	float max_val = -std::numeric_limits<float>::infinity();
                // 	for(int64_t q = 0; q < cache_length; ++q) {
                // 		max_val = std::max(max_val,
                // static_cast<float>(attn_weights[b * num_attention_heads * 1 *
                // cache_length + n * 1 * cache_length + q]));
                // 	}

                // 	// Calculate sum of exp(x - max_val)
                // 	float acc = 0.0f;
                // 	for(int64_t q = 0; q < cache_length; ++q) {
                // 		acc +=
                // std::exp(static_cast<float>(attn_weights[b
                // * num_attention_heads * 1 * cache_length + n * 1 *
                // cache_length + q]) - max_val);
                // 	}

                // 	// Normalize with the stabilized values
                // 	for(int64_t q = 0; q < cache_length; ++q) {
                // 		attn_weights[b * num_attention_heads * 1 *
                // cache_length
                // +
                // n
                // * 1 * cache_length + q] = c10::BFloat16(
                // 			std::exp(static_cast<float>(attn_weights[b
                // * num_attention_heads * 1 * cache_length + q]) - max_val) /
                // acc
                // 		);
                // 	}
                // }

                float* accumulator = static_cast<float*>(std::malloc(
                    num_attention_heads * head_dim * sizeof(float)));
                memset(accumulator, 0,
                       num_attention_heads * head_dim * sizeof(float));
                for (int64_t token = 0; token < cache_length; ++token) {
                    for (int64_t kv_head = 0; kv_head < num_kv_head;
                         ++kv_head) {
                        for (int64_t id_in_group = 0; id_in_group < group_size;
                             ++id_in_group) {
                            int64_t cur_attention_head =
                                kv_head * group_size + id_in_group;
                            float* out_ptr =
                                accumulator + cur_attention_head * head_dim;
                            c10::BFloat16* value_state_bf16 =
                                value_states[b] +
                                token * num_kv_head * head_dim +
                                kv_head * head_dim;
                            scalar_vector_multiply_add(
                                attn_weights[b * num_attention_heads * 1 *
                                                 cache_length +
                                             cur_attention_head * 1 *
                                                 cache_length +
                                             token],
                                value_state_bf16, out_ptr, head_dim);
                        }
                    }
                }
                for (int64_t i = 0; i < num_attention_heads * 1 * head_dim;
                     ++i) {
                    attn_output[b * num_attention_heads * 1 * head_dim + i] =
                        c10::BFloat16(accumulator[i]);
                }
                std::free(accumulator);
            }

            // // Softmax
            // #pragma omp for
            // for(int64_t b = 0; b < bsz; ++b){
            // 	for(int64_t n = 0; n < num_attention_heads; ++n){
            // 		float acc = 0.0f;
            // 		for(int64_t q = 0; q < cache_length; ++q){
            // 			acc +=
            // std::exp(static_cast<float>(attn_weights[b
            // * num_attention_heads * 1 * cache_length + n * 1 * cache_length +
            // q]));
            // 		}
            // 		for(int64_t q = 0; q < cache_length; ++q){
            // 			attn_weights[b * num_attention_heads * 1 *
            // cache_length + n * 1 * cache_length + q] = c10::BFloat16(
            // 				std::exp(static_cast<float>(attn_weights[b
            // * num_attention_heads * 1 * cache_length + n * 1 * cache_length +
            // q])) / acc
            // 			);
            // 		}
            // 	}
            // };

            // // Weighted sum of value_states
            // float* accumulator =
            // static_cast<float*>(std::malloc(num_attention_heads * head_dim *
            // sizeof(float))); #pragma omp for for(int64_t b = 0; b < bsz;
            // ++b){ 	memset(accumulator, 0, num_attention_heads * head_dim *
            // sizeof(float)); 	for(int64_t token = 0; token < cache_length;
            // ++token){ 		for(int64_t kv_head = 0; kv_head <
            // num_kv_head;
            // ++kv_head){ 			for(int64_t id_in_group = 0;
            // id_in_group < group_size;
            // ++id_in_group){ 				int64_t
            // cur_attention_head = kv_head * group_size + id_in_group;
            // float* out_ptr = accumulator + cur_attention_head * head_dim;
            // c10::BFloat16* value_state_bf16 = value_states[b] + token *
            // num_kv_head
            // * head_dim + kv_head * head_dim;
            // 				scalar_vector_multiply_add(attn_weights[b
            // * num_attention_heads * 1 * cache_length + cur_attention_head * 1
            // * cache_length + token], value_state_bf16, out_ptr, head_dim);
            // 			}
            // 		}
            // 	}
            // 	for(int64_t i = 0; i < num_attention_heads * 1 * head_dim; ++i){
            // 		attn_output[b * num_attention_heads * 1 * head_dim + i]
            // = c10::BFloat16(accumulator[i]);
            // 	}
            // }
            // std::free(accumulator);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        // std::cerr<<"grouped_query_attention_cpu_avx2 end parallel. Duration:
        // "<<duration.count()<<" ms."<<std::endl;

        auto option = torch::TensorOptions()
                          .dtype(torch::kBFloat16)
                          .device(torch::kCPU)
                          .requires_grad(false)
                          .memory_format(torch::MemoryFormat::Contiguous);
        torch::Tensor attn_output_tensor = torch::from_blob(
            attn_output, {bsz, num_attention_heads, 1, head_dim},
            [attn_output](void* p) { std::free(attn_output); }, option);

        // torch::Tensor attn_weights_tensor = torch::from_blob(attn_weights,
        // 													{bsz,
        // num_attention_heads, 1, cache_length}, [attn_weights](void* p)
        // {std::free(attn_weights);},
        // query_states.options());
        std::free(attn_weights);
        // std::free(accumulator);
        // std::free(attn_output);
        return attn_output_tensor;
    } catch (const std::exception& e) {
        std::cerr << "grouped_query_attention_cpu_avx2 error: " << e.what()
                  << std::endl;
        throw;
    } catch (...) {
        std::cerr << "grouped_query_attention_cpu_avx2 error: unknown error."
                  << std::endl;
        throw;
    }
};
