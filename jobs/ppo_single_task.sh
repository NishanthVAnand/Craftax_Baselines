#!/bin/bash

all_lrs1=(3e-4) # 8e-3 3e-3 1e-3 8e-4 3e-4 1e-4 8e-5 3e-5)
all_seeds=(0 1 2)

for lr1 in ${all_lrs1[@]}; do
	for seed in ${all_seeds[@]}; do
		sbatch run_ppo_single_task.sh $1 $lr1 $seed $2
	sleep 0.2;
	done
done