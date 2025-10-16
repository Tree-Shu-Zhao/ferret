# Search-R1

Search-R1 is a reinforcement learning framework designed for training reasoning-and-searching interleaved LLMs—language models that learn to reason and make tool calls (e.g., to search engines) in a coordinated manner.

## Usage

### Data Preparation

```bash
bash recipe/search-r1/data_preprocess_search_r1.sh
```

### Training

Train a Search-R1 model with Qwen2.5-3B:

```bash
bash recipe/search-r1/train_search-r1_ppo_qwen2.5-3b-instruct.sh
```

### Evaluation

```bash
bash recipe/search-r1/eval_search-r1.sh
```

## Model Checkpoints

| Model | Checkpoint |
|-------|----------------------|
| Qwen2.5-3B-Instruct | [Scout_Search-R1_Qwen2.5-3b-instruct_ppo](https://huggingface.co/TreezzZ/Scout_Search-R1_Qwen2.5-3b-instruct_ppo) |
| Qwen2.5-14B-Instruct | [Scout_Search-R1_Qwen2.5-14b-instruct_ppo](https://huggingface.co/TreezzZ/Scout_Search-R1_Qwen2.5-14b-instruct_ppo) |

## Links

- Paper: [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516) [An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved LLM Agents](https://arxiv.org/abs/2505.15117). We implement Search-R1 based on the second paper with a format score.
- Original Repo: [https://github.com/PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1)
