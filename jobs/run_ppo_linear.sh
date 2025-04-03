#!/bin/bash

#SBATCH --time=15:0:0
#SBATCH --mem=32GB
#SBATCH --gres=gpu:80GB:3
#SBATCH --cpus-per-task=8
#SBATCH --output=output/ppo_linear_%j.out
#SBATCH --error=output/ppo_linear_%j.err

module load python/3.9
module load cuda/12.6.0
source ~/crafter_jax/bin/activate
cd ..

echo "$@"
python ppo_llm.py --total_timesteps=$1 --num_envs=256 --lr=$2 --num_steps=16 --wandb_project="LLM-play" --wandb_entity="doina-precup" --seed=$3