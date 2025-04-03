#!/bin/bash

all_lrs1=(3e-5)
#all_lrs1=(3e-4 3e-6) sami will test these
all_seeds=(0 1 2)

for lr1 in ${all_lrs1[@]}; do
	for seed in ${all_seeds[@]}; do
		sbatch run_ppo_linear.sh $1 $lr1 $seed
	sleep 0.2;
	done
done