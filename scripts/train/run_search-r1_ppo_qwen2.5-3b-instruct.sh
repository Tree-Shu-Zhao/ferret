set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"

# Setup custom metrics (automatically detects metrics from reward function)
echo "Setting up custom metrics for automatic detection..."
bash "$PROJECT_DIR/scripts/setup_custom_metrics.sh"
DATA_DIR="$PROJECT_DIR/data"

CONFIG_PATH="$PROJECT_DIR/configs"
TOOL_CONFIG="$CONFIG_PATH/tools/search_tool_config.yaml"
REWARD_FUNCTION_PATH="$PROJECT_DIR/scout/reward_score"

BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
PROJECT_NAME="scout"
EXPERIMENT_NAME="search-r1_qwen2.5-3b-instruct-ppo"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH/train" \
    --config-name="ppo" \
    custom_reward_function.path="$REWARD_FUNCTION_PATH/search_r1_format.py" \
    custom_reward_function.name=compute_score_em \
    +custom_reward_function.reward_kwargs.structure_format_score=0.2 \
    +custom_reward_function.reward_kwargs.final_format_score=0.1 \
    +custom_reward_function.reward_kwargs.retrieval_score=0 \
    +custom_reward_function.reward_kwargs.format_score=0 \
    +custom_reward_function.reward_kwargs.score=1.0 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    critic.model.path=$BASE_MODEL \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    data.train_files="$DATA_DIR/search_r1_train.parquet" \
    data.val_files="$DATA_DIR/search_r1_test.parquet"  \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" $@
