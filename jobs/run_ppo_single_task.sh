#!/bin/bash

#SBATCH --time=0:15:0
#SBATCH --mem=32GB
#SBATCH --gres=gpu:40GB:1
#SBATCH --cpus-per-task=8
#SBATCH --output=output/ppo_single_task_%j.out
#SBATCH --error=output/ppo_single_task_%j.err

module load python/3.9
module load cuda/12.6.0
source ~/crafter_jax/bin/activate
cd ..

echo "$@"
python ppo.py --env_name="Craftax-Classic-Symbolic-v1" --total_timesteps=$1 --num_envs=256 --lr=$2 --num_steps=16 --wandb_project="LLM-play" --wandb_entity="doina-precup" --seed=$3 --achievement=$4