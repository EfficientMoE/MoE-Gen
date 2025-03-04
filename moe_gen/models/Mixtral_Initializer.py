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
import os
import time
import types
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig
from transformers.models.mixtral.modeling_mixtral import (
    MixtralForCausalLM,
    apply_rotary_pos_emb,
    repeat_kv,
)

from moe_gen.core_engine import MoE_Gen as core_engine
from moe_gen.engine import EngineConfig, ModelConfig
from moe_gen.models.Wrapper import Attn_Wrapper, Expert_Wrapper


def _QKV_Proj(self, hidden_states, position_ids, kv_seq_len):
    """
    Project the hidden states to query, key, value states.
    """
    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    # logging.debug(f"Entering rotary embedding")
    # print(f"Python Query states shape: {query_states.shape}", flush=True)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    # logging.debug(f"Exiting rotary embedding")

    return query_states, key_states, value_states


def _attn_QKT_softmax(self, query_states, key_states, attention_mask):
    """
    - Compute the attention weights.
    - Apply the causal mask.
    - Apply the softmax.
    """
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    attn_weights = torch.matmul(
        query_states, key_states.transpose(2, 3)
    ) / math.sqrt(self.head_dim)
    attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)
    attn_weights = nn.functional.dropout(
        attn_weights, p=self.attention_dropout, training=self.training
    )
    return attn_weights


def _attn_AV(self, attn_weights, value_states):
    """
    - A @ V
    """
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output


def _O_Proj(self, attn_output):
    """
    - Output projection.
    """
    bsz, _, q_len, _ = attn_output.size()
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    return attn_output


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


def decoding_attn(
    self,
    hidden_states,
    past_key_states,
    past_value_states,
    attention_mask,
    position_ids,
):
    # attention_mask = update_casual_mask(attention_mask)
    # Vanilla attn
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    new_key_states = self.k_proj(hidden_states)
    new_value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    new_key_states = new_key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    new_value_states = new_value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    kv_seq_len = attention_mask.shape[-1]
    # logging.debug(f"Entering rotary embedding, kv_seq_len: {kv_seq_len}")
    cos, sin = self.rotary_emb(new_value_states, seq_len=kv_seq_len)
    query_states, new_key_states = apply_rotary_pos_emb(
        query_states, new_key_states, cos, sin, position_ids
    )

    key_states = torch.cat([past_key_states, new_key_states], dim=2)
    value_states = torch.cat([past_value_states, new_value_states], dim=2)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(
        query_states, key_states.transpose(2, 3)
    ) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)
    attn_weights = nn.functional.dropout(
        attn_weights, p=self.attention_dropout, training=self.training
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, new_key_states, new_value_states


class Mixtral_Initializer:
    def __init__(
        self,
        huggingface_ckpt_name: str,
        hf_cache_dir: str,
        cache_dir: Optional[str],
        engine_config: EngineConfig,
    ):
        self.huggingface_ckpt_name = huggingface_ckpt_name
        self.cache_dir = cache_dir
        self.engine_config = engine_config

        self.model = None
        self.hf_model_config = AutoConfig.from_pretrained(
            huggingface_ckpt_name,
            cache_dir=hf_cache_dir,
            trust_remote_code=True,
        )
        # TODO:
        self.model_config = self._parse_model_config()
        self._default_engine_config()
        self.state_dict_name_map = {}
        self.weight_copy_task = {}
        self.pt_ckpt_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            self.huggingface_ckpt_name,
        )
        if not os.path.exists(self.pt_ckpt_dir):
            os.makedirs(self.pt_ckpt_dir)

    def _default_engine_config(self):
        if "mistralai/Mixtral-8x7B" in self.hf_model_config._name_or_path:
            self.engine_config.Module_Batching_Config.prefill_micro_batch_size = 50
            self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size = 64
            self.engine_config.Module_Batching_Config.expert_prefill_batch_size_upper_bound = 8192
            self.engine_config.Module_Batching_Config.expert_decoding_batch_size_upper_bound = 8192

            self.engine_config.Basic_Config.log_level = "info"
            # self.engine_config.Basic_Config.device = 7
            self.engine_config.Basic_Config.torch_dtype = torch.bfloat16
            self.engine_config.Basic_Config.dtype_str = "bfloat16"
            # self.engine_config.Basic_Config.device_torch = torch.device(f"cuda:{self.engine_config.Basic_Config.device}")
            mode = int(os.getenv("ATTN_MODE", 0))
            logging.info(f"Attention mode: {mode}")
            self.engine_config.Basic_Config.attn_mode = mode
            self.engine_config.Basic_Config.module_types = [
                "attn",
                "routed_expert",
            ]
            self.engine_config.Basic_Config.num_threads = 25
            # self.engine_config.Basic_Config.max_decoding_length = 512
            # self.engine_config.Basic_Config.padding_length = 32

            self.engine_config.GPU_Buffer_Config.num_prefill_module_buffer = {
                "attn": 0,
                "routed_expert": 7,
            }
            self.engine_config.GPU_Buffer_Config.num_decoding_module_buffer = {
                "attn": 0,
                "routed_expert": 7,
            }
            self.engine_config.GPU_Buffer_Config.num_k_buffer = 3
            self.engine_config.GPU_Buffer_Config.num_v_buffer = 3
            self.engine_config.GPU_Buffer_Config.kv_buffer_num_tokens = (
                self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size
                * (
                    self.engine_config.Basic_Config.padding_length
                    + self.engine_config.Basic_Config.max_decoding_length
                )
            )
            self.engine_config.GPU_Buffer_Config.module_shapes = {
                "attn": {
                    "q_proj.weight": [4096, 4096],
                    "k_proj.weight": [1024, 4096],
                    "v_proj.weight": [1024, 4096],
                    "o_proj.weight": [4096, 4096],
                },
                "routed_expert": {
                    "w1.weight": [14336, 4096],
                    "w2.weight": [4096, 14336],
                    "w3.weight": [14336, 4096],
                },
            }

            # self.engine_config.KV_Storage_Config.num_host_slots = 4551
            self.engine_config.KV_Storage_Config.reserved_length = (
                self.engine_config.Basic_Config.padding_length
                + self.engine_config.Basic_Config.max_decoding_length
            )
            self.engine_config.KV_Storage_Config.slot_byte_size = (
                self.engine_config.KV_Storage_Config.reserved_length
                * self.model_config.head_dim
                * self.model_config.num_key_value_heads
                * 2
            )
            self.engine_config.KV_Storage_Config.storage_byte_size = (
                self.engine_config.KV_Storage_Config.num_host_slots
                * self.engine_config.KV_Storage_Config.slot_byte_size
                * self.model_config.num_hidden_layers
            )

        elif "mistralai/Mixtral-8x22B" in self.hf_model_config._name_or_path:
            self.engine_config.Module_Batching_Config.prefill_micro_batch_size = 20
            self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size = 100
            self.engine_config.Module_Batching_Config.expert_prefill_batch_size_upper_bound = 8192
            self.engine_config.Module_Batching_Config.expert_decoding_batch_size_upper_bound = 8192

            self.engine_config.Basic_Config.log_level = "info"
            # self.engine_config.Basic_Config.device = 7
            self.engine_config.Basic_Config.torch_dtype = torch.bfloat16
            self.engine_config.Basic_Config.dtype_str = "bfloat16"
            # self.engine_config.Basic_Config.device_torch = torch.device("cuda:7")
            mode = int(os.getenv("ATTN_MODE", 0))
            logging.info(f"Attention mode: {mode}")
            self.engine_config.Basic_Config.attn_mode = mode
            self.engine_config.Basic_Config.module_types = [
                "attn",
                "routed_expert",
            ]
            self.engine_config.Basic_Config.num_threads = 25

            self.engine_config.GPU_Buffer_Config.num_prefill_module_buffer = {
                "attn": 0,
                "routed_expert": 8,
            }
            self.engine_config.GPU_Buffer_Config.num_decoding_module_buffer = {
                "attn": 0,
                "routed_expert": 8,
            }
            self.engine_config.GPU_Buffer_Config.num_k_buffer = 3
            self.engine_config.GPU_Buffer_Config.num_v_buffer = 3
            self.engine_config.GPU_Buffer_Config.kv_buffer_num_tokens = (
                self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size
                * (
                    self.engine_config.Basic_Config.padding_length
                    + self.engine_config.Basic_Config.max_decoding_length
                )
            )
            self.engine_config.GPU_Buffer_Config.module_shapes = {
                "attn": {
                    "q_proj.weight": [6144, 6144],
                    "k_proj.weight": [1024, 6144],
                    "v_proj.weight": [1024, 6144],
                    "o_proj.weight": [6144, 6144],
                },
                "routed_expert": {
                    "w1.weight": [16384, 6144],
                    "w2.weight": [6144, 16384],
                    "w3.weight": [16384, 6144],
                },
            }

            # self.engine_config.KV_Storage_Config.num_host_slots = 650
            self.engine_config.KV_Storage_Config.reserved_length = (
                self.engine_config.Basic_Config.padding_length
                + self.engine_config.Basic_Config.max_decoding_length
            )
            self.engine_config.KV_Storage_Config.slot_byte_size = (
                self.engine_config.KV_Storage_Config.reserved_length
                * self.model_config.head_dim
                * self.model_config.num_key_value_heads
                * 2
            )
            self.engine_config.KV_Storage_Config.storage_byte_size = (
                self.engine_config.KV_Storage_Config.num_host_slots
                * self.engine_config.KV_Storage_Config.slot_byte_size
                * self.model_config.num_hidden_layers
            )
        else:
            raise ValueError(
                f"{self.model.config._name_or_path} not supported yet."
            )

    def _parse_model_config(self):
        model_config = ModelConfig()
        if "mistralai/Mixtral-8x7B" in self.hf_model_config._name_or_path:
            model_config.model_type = "Mixtral"
            model_config.num_hidden_layers = 32
            model_config.num_local_experts = 8
            model_config.num_attention_heads = 32
            model_config.num_key_value_heads = 8
            model_config.hidden_size = 4096
            model_config.intermediate_size = 14336
            model_config.head_dim = 128

        elif "mistralai/Mixtral-8x22B" in self.hf_model_config._name_or_path:
            model_config.model_type = "Mixtral"
            model_config.num_hidden_layers = 56
            model_config.num_local_experts = 8
            model_config.num_attention_heads = 48
            model_config.num_key_value_heads = 8
            model_config.hidden_size = 6144
            model_config.intermediate_size = 16384
            model_config.head_dim = 128

        else:
            raise ValueError("Model not supported")

        return model_config

    def _save_safetensors_to_pt(self):
        ckpt_files = os.listdir(self.cache_dir)
        ckpt_files = [
            os.path.join(self.cache_dir, ckpt)
            for ckpt in ckpt_files
            if ckpt.endswith(".safetensors")
        ]

        from multiprocessing import Process

        from safetensors.torch import load_file

        def save_and_load(file_path, save_dir):
            tensor_dict = load_file(file_path)
            torch.save(tensor_dict, save_dir)
            return tensor_dict

        processes = []
        for ckpt in tqdm(
            ckpt_files, desc="Loading checkpoint files", smoothing=0
        ):
            dst_dir = os.path.join(
                self.pt_ckpt_dir,
                ckpt.split("/")[-1].replace(".safetensors", ".pt"),
            )
            if os.path.exists(dst_dir):
                continue

            p = Process(target=save_and_load, args=(ckpt, dst_dir))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            p.close()
        logging.info("All safetensor loader processes joined")
        # TODO:
        # if self.cache_dir is None:
        # 	from transformers import TRANSFORMERS_CACHE
        # 	self.cache_dir = TRANSFORMERS_CACHE
        # 	model_cache_dirs = os.listdir(self.cache_dir)
        # 	if "models--deepseek-ai-" + self.huggingface_ckpt_name not in model_cache_dirs:
        # 		tmp = DeepseekV2ForCausalLM.from_pretrained(self.huggingface_ckpt_name, torch_dtype="auto", trust_remote_code=True)
        # 		del tmp
        # 		gc.collect()
        # 	parameter_ckpt_dir = os.path.join(self.cache_dir, "models--deepseek-ai-" + self.huggingface_ckpt_name)
        # 	parameter_ckpt_dir = os.path.join(parameter_ckpt_dir, "snapshots")
        # 	parameter_ckpt_dir = os.listdir(parameter_ckpt_dir)[0]
        # 	ckpt_files = os.listdir(parameter_ckpt_dir)
        # 	ckpt_files = [os.path.join(parameter_ckpt_dir, ckpt) for ckpt in ckpt_files]

        # 	for ckpt in tqdm(
        # 		ckpt_files, desc="Loading checkpoint files", smoothing=0
        # 	):
        # 		tmp_state_dict = {}
        # 		if ckpt.endswith(".safetensors"):
        # 			with safe_open(ckpt, framework="pt", device="cpu") as f:
        # 				for k in f.keys():
        # 					tmp_state_dict[k] = f.get_tensor(k)
        # 			torch.save(tmp_state_dict, os.path.join(self.cus_ckpt_dir, ckpt.split("/")[-1].replace(".safetensors", ".pt")))

    def _load_model_skeleton(self, skieleton_state_dict):
        for key, param in skieleton_state_dict.items():
            levels = key.split(".")
            weight = self.model.__getattr__(levels[0])
            for level in levels[1:]:
                weight = weight.__getattr__(level)
            weight.data = param
        byte_size = (
            sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            * 2
            / (1024**3)
        )
        logging.info(f"Model size: {byte_size:.2f} GB")

    def Init(self):
        self._save_safetensors_to_pt()
        self._parse_state_dict()
        self.core_engine = core_engine(self.engine_config, self.model_config)
        self.core_engine.Init(
            self.pt_ckpt_dir, self.state_dict_name_map, self.weight_copy_task
        )
        skeleton_state_dict = self.core_engine.get_skeleton_state_dict()
        self._load_model_skeleton(skeleton_state_dict)
        self._config_attn_module()
        self._config_expert_module()
        self._config_lm_head_hook()

        self.model.to(self.engine_config.Basic_Config.device_torch)
        self.model.eval()
        print(self.model, flush=True)

        return (
            self.core_engine,
            self.model,
            self.engine_config,
            self.model_config,
        )

    def _lm_head_forward_pre_hook(self, module, input):
        return input[0][:, -1, :].unsqueeze(1)

    def _config_lm_head_hook(self):
        self.model.lm_head.register_forward_pre_hook(
            self._lm_head_forward_pre_hook
        )

    def _parse_state_dict(self):
        model_init_start_time = time.perf_counter()
        self.hf_model_config._attn_implementation = "flash_attention_2"
        self.model = MixtralForCausalLM._from_config(self.hf_model_config).to(
            self.engine_config.Basic_Config.device_torch
        )
        self.model.eval()
        logging.info(
            f"torch module init time: {time.perf_counter() - model_init_start_time} s"
        )

        self.weight_copy_task["attn"] = []
        self.weight_copy_task["routed_expert"] = []
        self.weight_copy_task["shared_expert"] = []

        # for layer_idx in trange(self.model_config.num_hidden_layers):
        # 	for name, _ in self.model.model.layers[layer_idx].self_attn.named_parameters():
        # 		tensor_full_name = "model.layers." + str(layer_idx) + ".self_attn." + name
        # 		self.state_dict_name_map[tensor_full_name] = {"module_key": "attn_" + str(layer_idx), "tensor_key": name}
        # 	self.weight_copy_task['attn'].append("attn_" + str(layer_idx))

        # 	for expert_idx in range(self.model_config.num_local_experts):
        # 		for name, _ in self.model.model.layers[layer_idx].block_sparse_moe.experts[expert_idx].named_parameters():
        # 			tensor_full_name = "model.layers." + str(layer_idx) + ".block_sparse_moe.experts." + str(expert_idx) + "." + name
        # 			self.state_dict_name_map[tensor_full_name] = {"module_key": "routed_expert_" + str(layer_idx) + "_" + str(expert_idx), "tensor_key": name}
        # 		self.weight_copy_task['routed_expert'].append("routed_expert_" + str(layer_idx) + "_" + str(expert_idx))

        # TODO:
        if "mistralai/Mixtral-8x22B" in self.hf_model_config._name_or_path:
            # Setting 2: Put Attn in the skeleton.
            for layer_idx in range(self.model_config.num_hidden_layers):
                for expert_idx in range(self.model_config.num_local_experts):
                    for name, _ in (
                        self.model.model.layers[layer_idx]
                        .block_sparse_moe.experts[expert_idx]
                        .named_parameters()
                    ):
                        tensor_full_name = (
                            "model.layers."
                            + str(layer_idx)
                            + ".block_sparse_moe.experts."
                            + str(expert_idx)
                            + "."
                            + name
                        )
                        self.state_dict_name_map[tensor_full_name] = {
                            "module_key": "routed_expert_"
                            + str(layer_idx)
                            + "_"
                            + str(expert_idx),
                            "tensor_key": name,
                        }
                    self.weight_copy_task["routed_expert"].append(
                        "routed_expert_"
                        + str(layer_idx)
                        + "_"
                        + str(expert_idx)
                    )

        if "mistralai/Mixtral-8x7B" in self.hf_model_config._name_or_path:
            # setting 2: Attn and first expert in the skeleton.
            for layer_idx in range(self.model_config.num_hidden_layers):
                for expert_idx in range(1, self.model_config.num_local_experts):
                    for name, _ in (
                        self.model.model.layers[layer_idx]
                        .block_sparse_moe.experts[expert_idx]
                        .named_parameters()
                    ):
                        tensor_full_name = (
                            "model.layers."
                            + str(layer_idx)
                            + ".block_sparse_moe.experts."
                            + str(expert_idx)
                            + "."
                            + name
                        )
                        self.state_dict_name_map[tensor_full_name] = {
                            "module_key": "routed_expert_"
                            + str(layer_idx)
                            + "_"
                            + str(expert_idx),
                            "tensor_key": name,
                        }
                    self.weight_copy_task["routed_expert"].append(
                        "routed_expert_"
                        + str(layer_idx)
                        + "_"
                        + str(expert_idx)
                    )

    def _config_attn_module(self):
        """
        - Configure the wrapper.
        """
        for layer_idx in range(len(self.model.model.layers)):
            attn_module = self.model.model.layers[layer_idx].self_attn
            setattr(
                attn_module,
                "QKV_Proj",
                types.MethodType(_QKV_Proj, attn_module),
            )
            setattr(
                attn_module,
                "attn_QKT_softmax",
                types.MethodType(_attn_QKT_softmax, attn_module),
            )
            setattr(
                attn_module, "attn_AV", types.MethodType(_attn_AV, attn_module)
            )
            setattr(
                attn_module, "O_Proj", types.MethodType(_O_Proj, attn_module)
            )
            setattr(
                attn_module,
                "decoding_attn",
                types.MethodType(decoding_attn, attn_module),
            )
            # setattr(attn_module, "decoding_flash", types.MethodType(decoding_flash, attn_module))
            # setattr(attn_module, "prefill_attn", types.MethodType(prefill_attn, attn_module))
            if "attn_" + str(layer_idx) in self.weight_copy_task["attn"]:
                get_weights = True
            else:
                get_weights = False
            attn_wrapper_instance = Attn_Wrapper(
                attn_module,
                layer_idx,
                self.core_engine,
                self.engine_config,
                self.model_config,
                get_weights,
            )
            self.model.model.layers[layer_idx].self_attn = attn_wrapper_instance

    def _config_expert_module(self):
        """
        Replace expert module with the wrapper.
        """
        for layer_idx in range(len(self.model.model.layers)):
            layer = self.model.model.layers[layer_idx]
            for expert_idx in range(len(layer.block_sparse_moe.experts)):
                expert_module = layer.block_sparse_moe.experts[expert_idx]
                if (
                    "routed_expert_" + str(layer_idx) + "_" + str(expert_idx)
                    in self.weight_copy_task["routed_expert"]
                ):
                    get_weights = True
                else:
                    get_weights = False
                expert_wrapper_instance = Expert_Wrapper(
                    expert_module,
                    layer_idx,
                    expert_idx,
                    self.core_engine,
                    self.engine_config,
                    self.model_config,
                    get_weights,
                )
                layer.block_sparse_moe.experts[expert_idx] = (
                    expert_wrapper_instance
                )
