#!/bin/bash
#SBATCH --job-name=arc-gemma4
#SBATCH --partition=agent-xlong
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=5-00:00:00
#SBATCH --output=gigpo_gemma4_%j.log

set -ex
export PATH=$HOME/.local/bin:$PATH
cd ~/verl-agent
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source ~/verl-venv/bin/activate
unset ROCR_VISIBLE_DEVICES

uv pip install -e . 2>&1 | tail -3
# Gemma 4 needs transformers >= 5.5.0 + compatible huggingface-hub
uv pip install "transformers>=5.5.0" "huggingface-hub>=1.10" --no-deps 2>&1 | tail -3

python3 -c 'import vllm; print("vllm:", vllm.__version__); import transformers; print("transformers:", transformers.__version__); import torch; print("torch:", torch.__version__)'

TRAIN_DATA=$HOME/data/verl-agent/arcagi3/train.parquet
VAL_DATA=$HOME/data/verl-agent/arcagi3/test.parquet

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=2 \
    data.val_batch_size=2 \
    data.max_prompt_length=8192 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.need_tools_kwargs=True \
    actor_rollout_ref.model.path=google/gemma-4-E4B-it \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=mean_norm \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.max_num_gen_batches=10 \
    env.env_name=ArcAgi3 \
    env.seed=0 \
    env.max_steps=1000 \
    env.rollout.n=4 \
    +env.arcagi3.env_dir=data/environment_files \
    +env.arcagi3.render_scale=4 \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=verl_agent_arcagi3 \
    trainer.experiment_name=gigpo_gemma4_e4b_tools \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=2 \
    trainer.total_epochs=200

echo "=== Done: $(date) ==="
