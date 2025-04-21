import subprocess
import os
import random

if __name__ == '__main__':

    temp_job_script_directory = "/scratch/s/saminur/"
    job_out_director = "/home/s/saminur/LLM_representation/wt_craftax-experiments-sami/Craftax_Baselines/jobs"

    os.makedirs(temp_job_script_directory, exist_ok=True)

    for achievement in ["PLACE_TABLE"]:
        for emb_type in [0]:
            for lr in [3e-4]:
                for seed in [0]:
                    for layer in [17]:

# emb_dict_map = {0: "mean", 1: "exp", 2: "last-10", 3: "last-k", 4: "eq-k", 5: "max", 6: "geom-k"}

                        python_run_command = f"python ppo_llm.py --total_timesteps=5e5 --num_envs=256 --lr={lr} --num_steps=16 --wandb_project='LLM-play' --wandb_entity='doina-precup' --seed={seed} --network_type='ActorCriticLinear' --layer={layer} --emb_type={emb_type} --eq_split=32 --env_name=Craftax-Classic-Symbolic-v1 --obs_type=2 --obs_only=2 --achievement=FULL"

                        job_script_content = f'''#!/bin/bash

#SBATCH --time=1:30:0
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

                        job_name = f"{achievement}-{seed}-{lr}-et_{emb_type}"
                        job_script_file_path = os.path.join(
                            temp_job_script_directory, f"{job_name}-{random.randint(0, 700)}.sh")

                        with open(job_script_file_path, 'w') as job_script_file:
                            job_script_file.write(job_script_content)

                        print(f'Launching {job_name}')
                        launch_command = f'sbatch --job-name={job_name} {job_script_file_path}'
                        subprocess.run(launch_command, shell=True,
                                       executable='/bin/bash')

                        os.remove(job_script_file_path)
