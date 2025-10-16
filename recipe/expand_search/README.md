# ExpandSearch

ExpandSearch goes beyond single-query limitations by training LLMs to perform iterative query expansion with reinforcement learning. The approach teaches models to refine and expand their search queries based on retrieved information, enabling more comprehensive information gathering.

## Usage

### Configure Squeezer Model

ExpandSearch uses a squeezer model powered by NVIDIA NIM API to refine and compress expanded queries. Before training, configure your NVIDIA API key:

```bash
# Get your API key from https://developer.nvidia.com/nim
export NVIDIA_API_KEY="your_nvidia_api_key_here"
```

**Important:** Training with the squeezer requires a large number of LLM API calls, which may incur significant costs. For cost-effective training, consider deploying your own LLM locally using a custom squeezer implementation (refer to `search_tool_with_squeezer_config.yaml` for configuration options).

### Data Preparation

```bash
bash recipe/expand_search/data_preprocess_expand_search.sh
```

### Training

Train an ExpandSearch model with Qwen2.5-3B:

```bash
bash recipe/expand_search/train_expand_search_ppo_qwen2.5-3b-instruct.sh
```

Note: ExpandSearch uses a specialized tool configuration with query squeezer (`search_tool_with_squeezer_config.yaml`) to refine expanded queries and compress information.

### Evaluation

```bash
bash recipe/expand_search/eval_ExpandSearch.sh
```

## Model Checkpoints

| Model | HuggingFace Checkpoint |
|-------|----------------------|
| Qwen2.5-3B-Instruct + Llama4-Maverick-17B (Squeezer) | [TreezzZ/Scout_ExpandSearch_Qwen2.5-3b-instruct_Llama4-Maverick-17b-128e-instruct_ppo](https://huggingface.co/TreezzZ/Scout_ExpandSearch_Qwen2.5-3b-instruct_Llama4-Maverick-17b-128e-instruct_ppo) |


## Links

- Project Page: [https://shuzhao.me/ExpandSearchProject/](https://shuzhao.me/ExpandSearchProject/)
- Paper: [Beyond the limitation of a single query: Train your LLM for query expansion with Reinforcement Learning (arXiv:2510.10009)](https://arxiv.org/abs/2510.10009)
