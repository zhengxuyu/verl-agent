#!/bin/bash
#SBATCH --job-name=gigpo_alfworld
#SBATCH --partition=agent-xlong
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=900G
#SBATCH --time=5-00:00:00
#SBATCH --nodelist=507-14
#SBATCH --output=%x_%j.log

cd /home/z84434016/verl-agent
bash examples/gigpo_dynamic_trainer/run_alfworld.sh vllm \
  algorithm.gigpo.history_replay.enable=True \
  algorithm.gigpo.history_replay.save_path=/home/fuyu/projects/residual_reward/verl-agent/history_alfworld.pkl \
  algorithm.gigpo.history_replay.load_path=/home/fuyu/projects/residual_reward/verl-agent/history_alfworld.pkl \
  algorithm.gigpo.history_replay.max_trajs_per_task=-1 \
  algorithm.gigpo.history_replay.max_saved_trajs_per_task=-1
