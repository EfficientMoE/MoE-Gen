<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/8e70c5d1-6c3a-4507-b3b2-821ebf127989">
    <img src="https://github.com/user-attachments/assets/5587e43e-a2ef-4dde-a84c-365c31f284f8" width=55%>
  </picture>
</p>


<div align="center">
 <h3>Extreme Throughput for Bulk Offline Inference for MoE Models with Limited GPU Resources</h3>
  <strong><a href="#Performance">Performance</a> | <a href="#Installation"> Installation</a> | <a href="#Quick-Start">Quick Start </a> </strong>
</div>

# Latest News
- [2025-03-21] Integrated mat absorption for DeepSeek MLA. On single A5000-24GB + 1TB host memory, **DeepSeek-R1-671B BF16 decoding: 27 token/s**. DeepSeek-V2-236B decoding: 49 tokens/s. On 4xA5000 , DeepSeek-V2-236B Prefill/Decoding: 3150/196 tokens/s.
- [2025-03] MoE-Gen V0.1 release. Empower **DeepSeek-R1-671B NO-Quantization** inference on **a single** NVIDIA A5000 with prefill throughput **204 tokens/s**, decoding throughput **17 tokens/s**. Data parallel on multiple devices supported.

# About
MoE-Gen is an efficient serving engine optimized specifically for **Mixture-of-Expert(MoE)** based large language models. It is tailored for bulk **offline inference** tasks and **limited GPU resources**. It enables low cost serving for latency-insensitive applications.

**Core Features**

- **Module-Based Batching**: A fine-grained batching strategy ensures consistently high GPU utilization throughout every forward pass.
- **Efficient Data Swapping Engine**: Supports inference of large-scale models (e.g., DeepSeek-R1) on constrained hardware setups such as single NVIDIA A5000 or RTX 4090 GPUs, aggressively maximizing overlap between computation and memory transfers to achieve optimal efficiency.
- **Heterogeneous Computing**: Leverages under-utilized CPU resources and optimizes host data locality to significantly boost overall throughput.

# Supported Models
- **DeepSeek-R1/V3-671B. FULL Precision.**
- DeepSeek-V2 Family: V2, V2.5, V2-Lite, V2-Coder.
- Mixtral: Mixtral 8x7B, Mixtral 8x22B.

# Application Scenarios
- MoE model evaluation.
- Company deployed LLM workflow for raw data formation.
- Latency-insensitive bulk inference tasks. Such as large batch inference launched in valley period.
- Deep-research applications. Deliver high-quality results overnight.

## Performance

**Single GPU A5000 (24GB Memory).** Prompt length 512 tokens, decoding length 256 tokens. Prefill/decoding throughput in tokens/s.


<div align="center">

|               | DeepSeek-R1/V3 (671B-W8A16)| DeekSeek-V2 (236B-W16A16)|Mixtral-8x22B (141B-W16A16)|Mixtral-8x7B (48B-W16A16)|
|:-------------:|:--------------------------:|:-----------------------:|:-----------------------:|:-----------------------:|
| **SGLang**    | Not Supported              | Not Supported           |Not Supported           |Not Supported           |
| **vLLM**      | Not Supported              | 97 / 0.9                |147/2                   |1347/31|
| **Llama.cpp** | 5.7/0.9                  | 23/1                    |110/2                   |328/4|
| **MoE-Gen**   | **204/27**                 | **787/49**              |**907/91**              |**2790/469**|
</div>


## Release Plan

* Integrate speculative decoding for DeepSeek-R1.
* Optimization on Ada and Hopper architecture.
* Support moonshotai/Moonlight MoE.

## Installation

We recommend installing MoE-Gen in a virtual environment. To install MoE-Gen, you can either install it from PyPI or build it from source.

### Hardware Requirements
```bash
Host Memory > Model Size
Disk Space >= 2 * Model Size
```
For example,  to efficiently serve DeepSeek-R1-6871B-FP8, 1TB Host memory is recommended. 1.5TB disk space is needed.


### Create conda environment.

```bash
conda create --name moe_gen python=3.11
conda activate moe_gen
```

### Install from PyPI

```bash
# install nightly release
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ moe-gen
pip install flash-attn==2.7.4.post1
```

## Quick Start
MoE-Gen seamlessly integrates with Huggingface environment. Start inference with Huggingface checkpoint name.
```python
import logging
from transformers import AutoTokenizer
from moe_gen.engine import moe_gen
import numpy as np
import datasets

if __name__ == "__main__":
    """
        Supported Models:
        1. DeepSeek-V2 Architecture:
            * deepseek-ai/DeepSeek-V2-Lite-Chat
            * deepseek-ai/DeepSeek-V2
            * deepseek-ai/DeepSeek-V2.5
            * deepseek-ai/DeepSeek-V2.5-1210
            * deepseek-ai/DeepSeek-Coder-V2-Instruct
        2. DeepSeek-V3 Architecture:
            * deepseek-ai/DeepSeek-R1
            * deepseek-ai/DeepSeek-V3
        3. Mixtral:
            * mistralai/Mixtral-8x7B-Instruct-v0.1
            * mistralai/Mixtral-8x22B-Instruct-v0.1
    """


    """
        Step 1: Select Model.
    """
    hugging_face_checkpoint = "deepseek-ai/DeepSeek-V2-Lite-Chat"

    """
        Step 2: Load dataset and apply chat template(prompt engineering).
    """

    benchmark_name = "Muennighoff/flan"
    dataset = datasets.load_dataset(benchmark_name, split="train")

    max_prompts = 500
    queries = dataset['inputs'][:max_prompts]

    tokenizer = AutoTokenizer.from_pretrained(hugging_face_checkpoint, trust_remote_code=True)
    for prompt_idx in range(len(queries)):
        messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": queries[prompt_idx]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        queries[prompt_idx] = text
    logging.info(f"Number of prompts: {len(queries)}")


    """
        Step 3: Launch MoE-Gen.
    """
    answer_set = moe_gen(
        huggingface_ckpt_name = hugging_face_checkpoint,
        queries=queries,
        max_input_length=512,
        max_decoding_length=32,
        device=[0], # Can also use multiple devices to enable data parallel, e.g.[0,1,2,3]
        host_kv_cache_size=10 #in GB, reserved CPU memory for KV-cache
    )


    """
        Step 4: Print responses to the prompts.
    """
    def decode_to_eos(tokenizer, tokens):
        tokens_array = np.array(tokens)
        eos_positions = np.where(tokens_array == tokenizer.eos_token_id)[0]
        end_pos = eos_positions[0] if len(eos_positions) > 0 else len(tokens_array)
        return tokenizer.decode(tokens[:end_pos], skip_special_tokens=True)

    print_result = True
    if print_result:
        for idx in range(len(answer_set)):
            tmp_answer = decode_to_eos(tokenizer, answer_set[idx].tolist()[0])
            print(f"Prompt {idx}: {queries[idx]}")
            print(f"Answer {idx}: {tmp_answer}")
            print("\n\n")



```
### Running Inference

This command runs the script on selected GPUs.
```bash
python example.py
```



## Citation
```
@misc{xu2025moegenhighthroughputmoeinference,
      title={MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching},
      author={Tairan Xu and Leyang Xue and Zhan Lu and Adrian Jackson and Luo Mai},
      year={2025},
      eprint={2503.09716},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2503.09716},
}
```
