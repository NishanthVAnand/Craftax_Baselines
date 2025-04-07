#!/bin/bash

all_lrs1=(3e-4)
#all_lrs1=(3e-4 3e-6) sami will test these
all_seeds=(0)

for lr1 in ${all_lrs1[@]}; do
	for seed in ${all_seeds[@]}; do
		sbatch run_ppo_NN.sh $1 $lr1 $seed $2
	sleep 0.2;
	done
done
