import subprocess
import os
import random

if __name__ == '__main__':

    temp_job_script_directory = "/scratch/s/saminur/"
    job_out_director = "/home/s/saminur/LLM_representation/wt_craftax-experiments-sami/Craftax_Baselines/jobs"

    os.makedirs(temp_job_script_directory, exist_ok=True)


    for emb_type in [0, 1, 4, 5]:
        for lr in [8e-4, 3e-4, 1e-4, 8e-5, 3e-5]:
            for seed in list(range(3)):

                python_run_command = f"python ppo_llm.py --total_timesteps=200000 --num_envs=256 --lr={lr} --num_steps=16 --wandb_project='LLM-play' --wandb_entity='doina-precup' --seed={seed} --network_type='ActorCriticLinear' --layer 17 --emb_type={emb_type} --eq_split=8 --env_name=Craftax-Classic-Symbolic-v1 --ent_coef=0.1 --obs_type=2 --obs_only=0 --achievement='PLACE_TABLE'"

                job_script_content = f'''#!/bin/bash

#SBATCH --time=3:0:0
#SBATCH --mem=32GB
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --output={job_out_director}/output/%j.out
#SBATCH --error={job_out_director}/output/%j.err

export WANDB_MODE=offline

source /home/s/saminur/LLM_representation/wt_craftax-experiments-sami/.venv/bin/activate
cd /home/s/saminur//LLM_representation/wt_craftax-experiments-sami/Craftax_Baselines/

python_command="{python_run_command}"
echo $python_command

eval $python_command
'''

                job_name = f"{seed}-{lr}-et_{emb_type}"
                job_script_file_path = os.path.join(
                    temp_job_script_directory, f"{job_name}-{random.randint(0, 700)}.sh")

                with open(job_script_file_path, 'w') as job_script_file:
                    job_script_file.write(job_script_content)

                launch_command = f'sbatch --job-name={job_name} {job_script_file_path}'
                subprocess.run(launch_command, shell=True,
                               executable='/bin/bash')

                # os.remove(job_script_file_path)
