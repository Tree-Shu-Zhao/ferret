# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Scout is an extensible framework for training LLM agents via reinforcement learning with advanced search and tool-use capabilities. It's built on top of VERL (Volcano Engine Reinforcement Learning for LLMs), implementing a Search-R1 like approach with multi-turn interactions and web search capabilities.

## Key Architecture Components

### Core Framework Integration
- **VERL**: The underlying RL training library providing flexible and efficient RLHF framework
- **SGLang**: Used for efficient rollout generation with multi-turn conversation support
- **FSDP**: Distributed training with Fully Sharded Data Parallel
- **Hydra**: Configuration management system for training parameters

### Main Components

1. **Data Processing Pipeline** (`scout/data/preprocess.py`):
   - Processes datasets from HuggingFace into Search-R1 format
   - Adds structured prompts with `<think>`, `<search>`, `<information>`, and `<answer>` tags
   - Configures tool usage kwargs for search functionality

2. **Reward System** (`scout/reward_score/`):
   - Custom reward functions for evaluating agent responses
   - Format validation for structured thinking patterns
   - Exact match (EM) scoring for answer evaluation
   - Configurable reward weights for different aspects (structure, retrieval, final answer)

3. **Retrieval Service** (`scout/retrieval/`):
   - Dense retrieval server using FAISS indexing
   - E5-based embedding model for semantic search
   - REST API for integration with training pipeline

4. **Tool Configuration** (`configs/tools/search_tool_config.yaml`):
   - Defines search tool schema and parameters
   - Configures retrieval service URL and rate limiting
   - Supports native and external tool integrations

## Common Development Commands

### Data Preparation
```bash
# Download and preprocess training data from HuggingFace
bash scripts/data_preprocess/data_preprocess.sh
# Or directly:
python scout/data/preprocess.py --local_dir data
```

### Training
```bash
# Run PPO training with Search-R1 configuration
bash scripts/train/run_search-r1_ppo_qwen2.5-3b-instruct.sh

# Key parameters to adjust:
# - data.train_batch_size: Training batch size (default: 512)
# - actor_rollout_ref.actor.optim.lr: Learning rate (default: 1e-6)
# - trainer.total_training_steps: Total steps (default: 1005)
# - trainer.n_gpus_per_node: Number of GPUs (default: 8)
```

### Evaluation
```bash
# Run evaluation on test set
bash scripts/eval/eval.sh

# For evaluation only (no training):
# Set trainer.val_only=True in the script
```

### Retrieval Server
```bash
# Start the retrieval service (required for training with search tools)
bash scripts/retrieval/run_retrieval_server.sh

# Requires:
# - Retrieval corpus at /mnt/data/retrieval-corpus/
# - FAISS index and Wikipedia corpus files
# - Runs on port 8000 by default
```

## Configuration Structure

The project uses Hydra for configuration management with hierarchical YAML files:

- `configs/train/`: Training configurations (PPO, Search-R1 PPO)
- `configs/tools/`: Tool configurations for search and retrieval
- Training configs inherit from VERL's default PPO trainer and override specific parameters

Key configuration groups:
- `algorithm`: RL algorithm settings (PPO, advantage estimation)
- `actor_rollout_ref`: Actor model and rollout generation settings
- `critic`: Critic model settings
- `data`: Dataset and batch configuration
- `trainer`: Training loop and logging settings
- `custom_reward_function`: Custom reward function paths and weights

## Multi-turn Conversation Format

The system uses Qwen format for multi-turn conversations with structured thinking:
1. System prompt defines agent capabilities
2. User query includes instructions for reasoning process
3. Assistant response follows pattern: `<think>` → `<tool_call>` → `<tool_response>` → repeat → `<answer>`

## Dependencies

Main dependencies (from `pyproject.toml`):
- Python >= 3.12
- PyTorch >= 2.8.0 with CUDA 12.9
- VERL (local editable installation)
- SGLang >= 0.5.3rc0
- Flash Attention
- WandB for experiment tracking

## Development Tips

1. **GPU Memory Management**: Adjust `actor_rollout_ref.rollout.gpu_memory_utilization` if encountering OOM errors
2. **Distributed Training**: Configure `trainer.n_gpus_per_node` and `trainer.nnodes` for multi-GPU/node setup
3. **Checkpointing**: Models are saved every `trainer.save_freq` steps
4. **Logging**: Use WandB for experiment tracking (set in `trainer.logger`)
5. **Custom Rewards**: Modify reward weights in training script's `custom_reward_function.reward_kwargs`

# Coding Guideline

1. You MUST NOT modify the verl folder since the scout codebase would like to seamlessly integrate with the latest verl.
2. You are ONLY ALLOWED to create new files if necessary. Try to reuse infra in verl.
3. You MUST NOT create any documentations like markdown files.
4. If you need to run test, please use the virtual environment in .venv
