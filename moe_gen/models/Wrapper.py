# ---------------------------------------------------------------------------- #
#  MoE-Gen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#                                                                               #
#  you may obtain a copy of the license at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
#                                                                               #
#  unless required by applicable law or agreed to in writing, software          #
#  distributed under the license is distributed on an "as is" basis,            #
#  without warranties or conditions of any kind, either express or implied.     #
#  see the license for the specific language governing permissions and          #
#  limitations under the license.                                               #
# ---------------------------------------------------------------------------- #

import logging
import math

import torch
import triton
import triton.language as tl
from transformers.cache_utils import DynamicCache


def update_casual_mask(attention_mask):
    dtype = torch.bfloat16
    min_dtype = torch.finfo(dtype).min
    device = attention_mask.device
    target_length = attention_mask.size(-1)
    sequence_length = 1
    cache_position = torch.arange(
        target_length - 1, target_length, device=device
    )
    causal_mask = torch.full(
        (sequence_length, target_length),
        fill_value=min_dtype,
        dtype=dtype,
        device=device,
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask *= torch.arange(
        target_length, device=device
    ) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(
        attention_mask.shape[0], 1, -1, -1
    )
    if attention_mask is not None:
        causal_mask = (
            causal_mask.clone()
        )  # copy to contiguous memory for in-place edit
        mask_length = attention_mask.shape[-1]
        padding_mask = (
            causal_mask[:, :, :, :mask_length]
            + attention_mask[:, None, None, :]
        )
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[
            :, :, :, :mask_length
        ].masked_fill(padding_mask, min_dtype)
    return causal_mask


def create_position_ids_from_attention_mask(
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    attention_mask: shape [batch_size, seq_len], with values in {0, 1}.
    Returns position_ids: same shape, where
      - tokens with attention_mask=0 get position_id=1
      - tokens with attention_mask=1 get a cumsum starting at 0
    """
    position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
    cumsum = torch.cumsum(attention_mask, dim=1) - 1
    position_ids[attention_mask == 1] = cumsum[attention_mask == 1]
    position_ids[attention_mask == 0] = 1
    return position_ids


# def deepseek_v3_dequantization(
# 		weight_data_fp8,
# 		weight_scale_inv_fp32,
# 		block_size = [128,128]) -> torch.Tensor:
# 	start_time = torch.cuda.Event(enable_timing=True)
# 	end_time = torch.cuda.Event(enable_timing=True)
# 	start_time.record()
# 	rows, cols = weight_data_fp8.size()
# 	n_block_rows = math.ceil(rows / block_size[0])
# 	n_block_cols = math.ceil(cols / block_size[1])
# 	assert n_block_cols == weight_scale_inv_fp32.size(1)
# 	assert n_block_rows == weight_scale_inv_fp32.size(0)

# 	dequantized_weight = weight_data_fp8.to(torch.float32)
# 	for i in range(n_block_rows):
# 		for j in range(n_block_cols):
# 			dequantized_weight[i*block_size[0]:(i+1)*block_size[0], j*block_size[1]:(j+1)*block_size[1]] *= weight_scale_inv_fp32[i, j]

# 	dequantized_weight = dequantized_weight.to(torch.bfloat16)
# 	end_time.record()
# 	torch.cuda.synchronize(7)
# 	# logging dequantize time in ms
# 	logging.info(f"Dequantization time: {start_time.elapsed_time(end_time)} ms")
# 	return dequantized_weight


def deepseek_v3_dequantization(
    weight_data_fp8, weight_scale_inv_fp32, block_size=[128, 128]
) -> torch.Tensor:
    # start_time = torch.cuda.Event(enable_timing=True)
    # end_time = torch.cuda.Event(enable_timing=True)
    # start_time.record()
    rows, cols = weight_data_fp8.size()
    n_block_rows = math.ceil(rows / block_size[0])
    n_block_cols = math.ceil(cols / block_size[1])
    assert n_block_cols == weight_scale_inv_fp32.size(1)
    assert n_block_rows == weight_scale_inv_fp32.size(0)

    dequantized_weight = weight_data_fp8.to(torch.float32)
    expanded_scales = weight_scale_inv_fp32.repeat_interleave(
        block_size[0], dim=0
    ).repeat_interleave(block_size[1], dim=1)
    expanded_scales = expanded_scales[
        :rows, :cols
    ]  # trim if block doesn't perfectly divide
    dequantized_weight *= expanded_scales
    # for i in range(n_block_rows):
    # 	for j in range(n_block_cols):
    # 		dequantized_weight[i*block_size[0]:(i+1)*block_size[0], j*block_size[1]:(j+1)*block_size[1]] *= weight_scale_inv_fp32[i, j]

    dequantized_weight = dequantized_weight.to(torch.bfloat16)
    # end_time.record()
    # torch.cuda.synchronize(7)
    # # logging dequantize time in ms
    # logging.info(f"Dequantization time: {start_time.elapsed_time(end_time)} ms")
    return dequantized_weight


# def deepseek_v3_dequantization(
#     weight_data_fp8: torch.Tensor,
#     weight_scale_inv_fp32: torch.Tensor,
#     block_size=(128, 128)
# ) -> torch.Tensor:
#     """
#     Vectorized dequantization that removes Python-level loops
#     and leverages PyTorch's parallelism.
#     """
#     rows, cols = weight_data_fp8.shape
#     block_rows, block_cols = block_size

#     # Number of blocks in each dimension
#     n_block_rows = rows // block_rows
#     n_block_cols = cols // block_cols

#     # 1) Reshape weight data into 4D block form and cast to float32
#     #    shape becomes [n_block_rows, block_rows, n_block_cols, block_cols].
#     weight_4d = weight_data_fp8.reshape(n_block_rows, block_rows, n_block_cols, block_cols).to(torch.float32)

#     # 2) Broadcast scale into 4D by unsqueezing along the second and fourth dimensions.
#     #    shape becomes [n_block_rows, 1, n_block_cols, 1].
#     scale_4d = weight_scale_inv_fp32.unsqueeze(1).unsqueeze(-1)

#     # 3) Multiply once using broadcasting
#     dequantized_4d = weight_4d * scale_4d

#     # 4) Reshape back to [rows, cols] and cast to bfloat16
#     dequantized_weight = dequantized_4d.reshape(rows, cols).to(torch.bfloat16)

#     return dequantized_weight


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes weights using the provided scaling factors and stores the result.

    Args:
            x_ptr (tl.pointer): Pointer to the quantized weights.
            s_ptr (tl.pointer): Pointer to the scaling factors.
            y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
            M (int): Number of rows in the weight matrix.
            N (int): Number of columns in the weight matrix.
            BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
            None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
            x (torch.Tensor): The quantized weight tensor of shape (M, N).
            s (torch.Tensor): The scale tensor of shape (M, N).
            block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
            torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
            AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert (
        x.is_contiguous() and s.is_contiguous()
    ), "Input tensors must be contiguous"
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.bfloat16)
    grid = lambda meta: (  # noqa E731
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


class Attn_Wrapper(torch.nn.Module):
    phase = "prefill"
    attn_mode = 0
    cur_batch = None

    def __init__(
        self,
        module,
        layer_idx,
        core_engine,
        engine_config,
        model_config,
        get_weights,
        weight_dequant_scale=None,
    ):
        super().__init__()
        self.module = module
        self.layer_idx = layer_idx
        self.core_engine = core_engine
        self.engine_config = engine_config
        self.model_config = model_config
        self.get_weights = get_weights
        self.attn_module_id = "attn" + "_" + str(self.layer_idx)
        self.weight_dequant_scale = weight_dequant_scale

    def forward(self, *args, **kwargs):
        logging.debug(
            f"[Layer {self.layer_idx} - Attn_Wrapper] Forward pass. Phase: {Attn_Wrapper.phase}"
        )
        self.core_engine.clear_expert_buffer(self.layer_idx, 0)
        # Step 1: Synchronize Attn Weights.
        if self.get_weights:
            weights_dict = self.core_engine.get_weights(self.attn_module_id)
            for name, param in self.module.named_parameters():
                if (
                    self.weight_dequant_scale is not None
                    and name + "_scale_inv" in self.weight_dequant_scale
                ):
                    param.data = deepseek_v3_dequantization(
                        weights_dict[name],
                        self.weight_dequant_scale[name + "_scale_inv"],
                    )
                else:
                    param.data = weights_dict[name]
        if Attn_Wrapper.phase == "prefill":
            """
				All attn Mode has the same prefill logic.
			"""
            # Get all tensor objects currently alive
            # for obj in gc.get_objects():
            # 	try:
            # 		if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            # 			if obj.numel() * obj.element_size() * 2 > 1e8:
            # 				if obj.device.type == "cuda":
            # 					logging.debug(f"Tensor object id: {id(obj)},size: {obj.numel() * obj.element_size() * 2}, shape: {obj.shape}")
            # 	except:
            # 		pass
            # Step 2: Prepare input
            arg_dict = {
                "hidden_states": kwargs["hidden_states"],
                "attention_mask": kwargs["attention_mask"],
                "position_ids": kwargs["position_ids"],
                "past_key_value": kwargs["past_key_value"],
                "output_attentions": kwargs["output_attentions"],
                "use_cache": kwargs["use_cache"],
            }
            past_key_value = DynamicCache()
            for i in range(self.layer_idx):
                past_key_value.key_cache.append(None)
                past_key_value.value_cache.append(None)
            arg_dict["past_key_value"] = past_key_value

            # Step 3: Forward pass
            num_prefill_batch = math.ceil(
                len(arg_dict["hidden_states"])
                / self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size
            )
            attn_output = torch.zeros_like(arg_dict["hidden_states"])
            for prefill_attn_batch_idx in range(num_prefill_batch):
                cur_batch_start = (
                    prefill_attn_batch_idx
                    * self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size
                )
                cur_batch_end = (
                    len(arg_dict["hidden_states"])
                    if (prefill_attn_batch_idx + 1)
                    * self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size
                    > len(arg_dict["hidden_states"])
                    else (prefill_attn_batch_idx + 1)
                    * self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size
                )
                cur_attn_batch = Attn_Wrapper.cur_batch[
                    cur_batch_start:cur_batch_end
                ]
                cur_hidden_states = arg_dict["hidden_states"][
                    cur_batch_start:cur_batch_end
                ]
                if "deepseek" in self.model_config.model_type:
                    cur_attention_mask = arg_dict["attention_mask"][
                        cur_batch_start:cur_batch_end
                    ]
                else:
                    cur_attention_mask = Attn_Wrapper.attention_mask[
                        cur_batch_start:cur_batch_end
                    ]

                key_cache = None
                value_cache = None
                with torch.no_grad():
                    if "deepseek" in self.model_config.model_type:
                        position_ids = Attn_Wrapper.position_ids[
                            cur_batch_start:cur_batch_end
                        ]
                        # output = self.module.prefill_attn_fp8(
                        #     cur_hidden_states, cur_attention_mask, position_ids,
                        #     self.weight_dequant_scale
                        # )
                        output = self.module.prefill_attn(
                            cur_hidden_states,
                            cur_attention_mask,
                            position_ids,
                        )
                        key_cache = output[1]
                        value_cache = torch.ones(
                            1,
                            dtype=torch.bfloat16,
                            device=kwargs["hidden_states"].device,
                        )
                    else:
                        output = self.module(**arg_dict)
                        key_cache = output[2].key_cache[self.layer_idx]
                        value_cache = output[2].value_cache[self.layer_idx]

                    torch.cuda.current_stream().synchronize()
                    attn_output[cur_batch_start:cur_batch_end] = output[0]

                self.core_engine.kv_offload(
                    self.layer_idx, cur_attn_batch, key_cache, value_cache
                )

            # Step 4: Clean up
            if self.get_weights:
                self.core_engine.free_weights_buffer(self.attn_module_id)
                for name, param in self.module.named_parameters():
                    param.data = torch.tensor(
                        0.0, dtype=param.data.dtype, device=param.data.device
                    )

            logging.debug(
                f"[Layer {self.layer_idx} - Attn_Wrapper] Finish forward pass. Phase: {Attn_Wrapper.phase}"
            )

            return attn_output, None, None

        elif Attn_Wrapper.phase == "decoding":
            self.core_engine.clear_expert_buffer(self.layer_idx, 0)

            hidden_states = kwargs["hidden_states"]
            attention_mask = Attn_Wrapper.attention_mask
            position_ids = Attn_Wrapper.position_ids
            final_attn_result = self.core_engine.attn(
                self.module,
                self.layer_idx,
                hidden_states,
                attention_mask,
                position_ids.to(self.engine_config.Basic_Config.device_torch),
                Attn_Wrapper.cur_batch,
            )
            torch.cuda.current_stream().synchronize()
            # Step 4: Clean up
            if self.get_weights:
                self.core_engine.free_weights_buffer(self.attn_module_id)
                for name, param in self.module.named_parameters():
                    param.data = torch.tensor(
                        0.0, dtype=param.data.dtype, device=param.data.device
                    )

            logging.debug(
                f"[Layer {self.layer_idx} - Attn_Wrapper] Finish forward pass. Phase: {Attn_Wrapper.phase}"
            )
            return final_attn_result, None, None


class Expert_Wrapper(torch.nn.Module):
    """
    For Mixtral and Qwen2MoE with transformers==4.42.0
    """

    phase = "prefill"

    def __init__(
        self,
        expert_module,
        layer_idx,
        expert_idx,
        core_engine,
        engine_config,
        model_config,
        get_weights,
        weight_dequant_scale=None,
    ):
        super().__init__()
        self.module = expert_module
        self.layer_idx = layer_idx
        self.expert_idx = expert_idx
        self.engine_config = engine_config
        self.model_config = model_config
        self.core_engine = core_engine
        self.get_weights = get_weights
        self.weight_dequant_scale = weight_dequant_scale
        if self.expert_idx >= 0:
            self.expert_weights_idx = (
                "routed_expert"
                + "_"
                + str(self.layer_idx)
                + "_"
                + str(self.expert_idx)
            )
        else:
            self.expert_weights_idx = (
                "shared_expert" + "_" + str(self.layer_idx)
            )

    def forward(self, *args, **kwargs):
        logging.debug(
            f"[Layer {self.layer_idx} - Expert {self.expert_idx}] Forward pass. Phase: {Expert_Wrapper.phase}"
        )
        # Step 1: Synchronize Expert Weights.
        if self.expert_idx != -1:
            self.core_engine.clear_expert_buffer(
                self.layer_idx, self.expert_idx
            )
        if self.get_weights:
            weights_dict = self.core_engine.get_weights(self.expert_weights_idx)
            for name, param in self.module.named_parameters():
                if (
                    self.weight_dequant_scale is not None
                    and name + "_scale_inv" in self.weight_dequant_scale
                ):
                    param.data = deepseek_v3_dequantization(
                        weights_dict[name],
                        self.weight_dequant_scale[name + "_scale_inv"],
                    )
                else:
                    param.data = weights_dict[name]

        # Step 2: Forward pass, micro-batching in case of OOM.
        hidden_states = args[0]
        result = torch.zeros_like(hidden_states)
        token_num_upper_bound = (
            self.engine_config.Module_Batching_Config.expert_prefill_batch_size_upper_bound
            if Expert_Wrapper.phase == "prefill"
            else self.engine_config.Module_Batching_Config.expert_decoding_batch_size_upper_bound
        )
        num_micro_batch = math.ceil(len(hidden_states) / token_num_upper_bound)
        for i in range(num_micro_batch):
            start = i * token_num_upper_bound
            end = (
                len(hidden_states)
                if (i + 1) * token_num_upper_bound > len(hidden_states)
                else (i + 1) * token_num_upper_bound
            )
            micro_batch = hidden_states[start:end]
            # self.module.eval()
            with torch.no_grad():
                result[start:end].copy_(self.module(micro_batch))
            torch.cuda.current_stream().synchronize()

        # Step 3: Clean up
        if self.get_weights:
            self.core_engine.free_weights_buffer(self.expert_weights_idx)
            for name, param in self.module.named_parameters():
                param.data = torch.tensor(
                    0.0, dtype=param.data.dtype, device=param.data.device
                )

        logging.debug(
            f"[Layer {self.layer_idx} - Expert {self.expert_idx}] Finish forward pass. Phase: {Expert_Wrapper.phase}"
        )
        if self.expert_idx != -1:
            self.core_engine.clear_expert_buffer(
                self.layer_idx, self.expert_idx
            )
        return result
