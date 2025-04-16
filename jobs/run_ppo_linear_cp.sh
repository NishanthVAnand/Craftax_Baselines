#!/bin/bash

#SBATCH --time=3:0:0
#SBATCH --mem=32GB
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --output=output/ppo_linear_%j.out
#SBATCH --error=output/ppo_linear_%j.err

export WANDB_MODE=offline


seed=$1

source ~/LLM_representation/wt_craftax-experiments-sami/.venv/bin/activate
cd ~/LLM_representation/wt_craftax-experiments-sami/Craftax_Baselines/


python_command="python ppo_llm.py --total_timesteps=200000 --num_envs=256 --lr=1e-4 --num_steps=16 --wandb_project='LLM-play' --wandb_entity='doina-precup' --seed=$seed --network_type='ActorCriticLinear' --layer 17 --emb_type=0 --eq_split=8 --env_name=Craftax-Classic-Symbolic-v1 --ent_coef=0.1 --obs_type=2 --obs_only=0 --achievement='PLACE_TABLE'"

eval $python_command
