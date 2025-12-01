# Ferret: Extensible RL Framework for Training Search Agents

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2B-red.svg)](https://pytorch.org/)

Ferret is an extensible framework for training Large Language Model (LLM) agents via reinforcement learning with advanced search capability. Built on top of [VERL](https://github.com/volcengine/verl), Ferret implements state-of-the-art search strategies with multi-turn interactions and retrieval capabilities. Ferret is designed to seamlessly integrate with the latest VERL releases without requiring framework modifications, and supports cutting-edge LLMs including Qwen3 and other state-of-the-art models.

## News

- [10/24/2025] Ferret now integrates VERL v0.7.0-dev, featuring fully asynchronous rollout mode and token-in-token-out for improved stability and efficiency in multi-turn RL training.
- [10/12/2025] We have released [Ferret](https://github.com/Tree-Shu-Zhao/ferret), an extensible RL framework for training LLM agents with advanced search capabilities, built on VERL and supporting state-of-the-art search strategies.

## 🌟 Key Features

- **🔍 Multiple Search Strategies**: Implements ParallelSearch, ExpandSearch, and Search-R1 with more strategies in active development
- **🛠️ Modular Architecture**: Easy-to-customize components for data, rewards, and tools
- **🤖 Latest LLM Support**: Compatible with cutting-edge models including Qwen3, Qwen2.5, and other state-of-the-art LLMs with the latest VERL framework
- **🚀 High-Performance Training**: Efficient RLHF framework with FSDP distributed training

## 📋 Table of Contents

- [Key Features](#-key-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Recipe Examples](#-recipe-examples)
- [Model Checkpoints](#-model-checkpoints)
- [Customization Guide for Researchers](#-customization-guide-for-researchers)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)

## 🔧 Installation

### Prerequisites

- Python >= 3.12
- CUDA 12.9+ compatible GPU
- 8+ GPUs recommended for training (configurable)
- **Recommended Docker Environment**: `nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04` for optimal compatibility

### Step 1: Clone the Repository

```bash
git clone https://github.com/Tree-Shu-Zhao/ferret.git
cd ferret
git submodule update --init --recursive
```

### Step 2: Set Up Virtual Environment

First, install uv by following the instructions at https://docs.astral.sh/uv/#installation if you haven't already.

```bash
# Using uv (recommended)
uv sync
source .venv/bin/activate
```

## 🚀 Quick Start

### 1. Prepare Training Data

Download and preprocess data:

```bash
bash recipe/search-r1/data_preprocess_search_r1.sh
```

### 2. Set Up and Start Retrieval Service (Required for Training)

The retrieval service provides semantic search capabilities for the training pipeline. Follow these steps to set it up:

#### Prerequisites Setup

```bash
# Set data directory (adjust path as needed)
export DATA_DIR="/mnt/data/retrieval-corpus"

# Create and activate conda environment
conda create -n retriever python=3.10 -y
conda activate retriever

# Install PyTorch with CUDA support
conda install pytorch==2.4.0 pytorch-cuda=12.6 -c pytorch -c nvidia -y

# Install FAISS for efficient similarity search
conda install -c pytorch -c nvidia faiss-gpu=1.8.0 -y

# Install Python dependencies
pip install transformers datasets pyserini
pip install uvicorn fastapi
pip install scipy==1.11.2
```

#### Download Retrieval Data

```bash
# Download the E5 index and Wikipedia corpus from HuggingFace
hf download --repo-type dataset --local-dir $DATA_DIR PeterJinGo/wiki-18-e5-index
hf download --repo-type dataset --local-dir $DATA_DIR PeterJinGo/wiki-18-corpus

# Prepare the index and corpus files
cd $DATA_DIR
cat part_* > e5_Flat.index
gzip -d wiki-18.jsonl.gz
```

#### Start the Service

```bash
# Run the retrieval server
bash scripts/retrieval/run_retrieval_server.sh
```

The service will run on port 8000 by default. Ensure the service is running before starting training.

### 3. Train Your First Model

```bash
bash recipe/search-r1/train_search-r1_ppo_qwen2.5-3b-instruct.sh

# To use Qwen3 or other latest models, simply modify the model path:
# Edit the script and change: actor_rollout_ref.model.path="Qwen/Qwen3-4B-Instruct"
```

### 4. Convert Checkpoints to HuggingFace Format Model

```bash
python -m verl.model_merger merge --backend fsdp --local_dir checkpoints/ferret/search-r1_qwen2.5-3b-instruct-ppo/global_step_300/actor --target_dir hf_models/Ferret_Search-R1_Qwen2.5-3b-instruct_ppo
```

### 5. Evaluate the Model

```bash
bash recipe/search-r1/eval_search-r1.sh
```

## 📚 Recipe Examples

The `recipe/` folder contains complete end-to-end examples for different search strategies. Each recipe includes data preprocessing, training, and evaluation scripts.

Each recipe folder provides:
- Data preprocessing scripts for the specific strategy
- Training scripts with recommended hyperparameters
- Evaluation scripts for benchmarking
- Strategy-specific configuration examples

Browse the `recipe/` folder to find the search strategy that best fits your research needs, or use them as templates to create your own custom strategies.

## 🤗 Model Checkpoints

Access our models on Hugging Face:

[🔗 Ferret Collection](https://huggingface.co/collections/TreezzZ/ferret-68ec65519567237a52d16627)

## 🔬 Customization Guide for Researchers

Ferret is designed to be easily extensible for research purposes. Here's how to add your own components:

### Adding Custom Prompt Templates

Create a new template in `ferret/data/templates/`:

```python
# ferret/data/templates/your_strategy.py
from . import PromptTemplate, register_template

TEMPLATE_NAME = "your_strategy"
DESCRIPTION = "Description of your search strategy"

SYSTEM_CONTENT = "You are a helpful AI assistant with search capabilities."

USER_CONTENT_PREFIX = """Answer the given question.
You must reason step by step in <think> tags.
Use <tool_call> to search for information.
Provide your final answer in <answer> tags.
Question: """

# Register the template
template = PromptTemplate(
    name=TEMPLATE_NAME,
    description=DESCRIPTION,
    system_content=SYSTEM_CONTENT,
    user_content_prefix=USER_CONTENT_PREFIX
)
register_template(template)
```

Then use it in preprocessing:

```bash
python ferret/data/preprocess.py --template_name your_strategy
```

### Implementing Custom Reward Functions

Add a new reward function in `ferret/reward_score/`:

```python
# ferret/reward_score/your_reward.py
import re

def compute_score_custom(solution_str, ground_truth, data_source, extra_info, **kwargs):
    """
    Custom reward function for your search strategy.

    Args:
        solution_str: The model's response
        ground_truth: Expected answer(s)
        data_source: Dataset source identifier
        extra_info: Additional metadata
        **kwargs: Configurable reward weights

    Returns:
        dict: {'reward_tensor': reward_value, 'reward_extra_info': metrics}
    """
    # Extract configurable weights
    format_score = kwargs.get('format_score', 0.2)
    retrieval_score = kwargs.get('retrieval_score', 0.3)
    answer_score = kwargs.get('answer_score', 1.0)

    # Validate format
    has_valid_format = check_format(solution_str)

    # Check retrieval quality
    retrieval_quality = evaluate_retrieval(solution_str, ground_truth)

    # Check answer correctness
    answer_correct = check_answer(solution_str, ground_truth)

    # Calculate final reward
    reward = 0
    if has_valid_format:
        reward += format_score
    if retrieval_quality > 0.5:
        reward += retrieval_score
    if answer_correct:
        reward = answer_score

    # Return detailed metrics for logging
    metrics = {
        "format_valid": has_valid_format,
        "retrieval_quality": retrieval_quality,
        "answer_correct": answer_correct,
        "final_reward": reward
    }

    return {
        "reward_tensor": reward,
        "reward_extra_info": metrics
    }
```

Configure in training script:

```bash
python3 -m verl.trainer.main_ppo \
    custom_reward_function.path="ferret/reward_score/your_reward.py" \
    custom_reward_function.name=compute_score_custom \
    +custom_reward_function.reward_kwargs.format_score=0.2 \
    +custom_reward_function.reward_kwargs.retrieval_score=0.3 \
    +custom_reward_function.reward_kwargs.answer_score=1.0
```

### Extending Tool Capabilities

Add new tools by creating a configuration in `configs/tools/`:

```yaml
# configs/tools/your_tool_config.yaml
tools:
  - class_name: verl.tools.your_tool.YourTool
    config:
      api_url: http://your-service:8000/api
      timeout: 30
      rate_limit: 100
    tool_schema:
      type: function
      function:
        name: your_tool
        description: Description of what your tool does
        parameters:
          type: object
          properties:
            param1:
              type: string
              description: Description of parameter 1
            param2:
              type: array
              items:
                type: string
              description: Description of parameter 2
          required:
            - param1
```

### Creating New Search Strategies

Combine templates, rewards, and tools to create a new strategy:

1. **Define the template** (ferret/data/templates/)
2. **Implement the reward function** (ferret/reward_score/)
3. **Configure tools** (configs/tools/)
4. **Create a recipe folder with training scripts** (recipe/your_strategy/)

Example training script for a new strategy:

```bash
#!/bin/bash
# recipe/your_strategy/train_your_strategy.sh

PROJECT_DIR="$(pwd)"
DATA_DIR="$PROJECT_DIR/data"
CONFIG_PATH="$PROJECT_DIR/configs"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH/train" \
    --config-name="ppo" \
    custom_reward_function.path="$PROJECT_DIR/ferret/reward_score/your_reward.py" \
    custom_reward_function.name=compute_score_custom \
    actor_rollout_ref.model.path="Qwen/Qwen2.5-3B-Instruct" \
    data.train_files="$DATA_DIR/your_strategy_train.parquet" \
    data.val_files="$DATA_DIR/your_strategy_test.parquet" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$CONFIG_PATH/tools/your_tool_config.yaml"
```

## 🙏 Acknowledgments

Ferret is built on top of several excellent projects:
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1)
- [VERL](https://github.com/volcengine/verl)
- [SGLang](https://github.com/sgl-project/sglang)

---

⭐ Star this repo if you find it useful!
