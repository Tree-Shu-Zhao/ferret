# ParallelSearch

ParallelSearch is a novel search strategy that trains LLMs to recognize parallelizable query structures and execute multiple search operations concurrently. By decomposing complex questions into independent sub-queries, ParallelSearch significantly improves retrieval efficiency and answer quality.

## Usage

### Data Preparation

```bash
bash recipe/parallel_search/data_preprocess_parallel_search.sh
```

### Training

Train ParallelSearch models with different base models:

```bash
# Qwen2.5-7B
bash recipe/parallel_search/train_parallel_search_ppo_qwen2.5-7b-instruct.sh

# Qwen3-4B
bash recipe/parallel_search/train_parallel_search_ppo_qwen3-4b-instruct.sh

# Qwen3-30B-A3B
bash recipe/parallel_search/train_parallel_search_ppo_qwen3-30b-a3b-instruct.sh
```

### Evaluation

```bash
bash recipe/parallel_search/eval_ParallelSearch.sh
```

## Model Checkpoints

| Model | HuggingFace Checkpoint |
|-------|----------------------|
| Qwen2.5-7B-Instruct | [TreezzZ/Ferret_ParallelSearch_Qwen2.5-7b-instruct_ppo](https://huggingface.co/TreezzZ/Ferret_ParallelSearch_Qwen2.5-7b-instruct_ppo) |
| Qwen3-4B-Instruct | [TreezzZ/Ferret_ParallelSearch_Qwen3-4b-instruct_ppo](https://huggingface.co/TreezzZ/Ferret_ParallelSearch_Qwen3-4b-instruct_ppo) |
| Qwen3-30B-A3B-Instruct | [TreezzZ/Ferret_ParallelSearch_Qwen3-30b-a3b-instruct_ppo](https://huggingface.co/TreezzZ/Ferret_ParallelSearch_Qwen3-30b-a3b-instruct_ppo) |

## Links

- Project Page: [https://shuzhao.me/ParallelSearchProject/](https://shuzhao.me/ParallelSearchProject/)
- Paper: [ParallelSearch: Train your LLMs to Decompose Query and Search Sub-queries in Parallel with Reinforcement Learning (arXiv:2508.09303)](https://arxiv.org/abs/2508.09303)
- Original Repo: [https://github.com/Tree-Shu-Zhao/ParallelSearch](https://github.com/Tree-Shu-Zhao/ParallelSearch)
