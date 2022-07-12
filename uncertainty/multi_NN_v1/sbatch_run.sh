#!/bin/bash -l
#SBATCH --job-name=ipython-trial
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task 12
#SBATCH --mem 80000
#SBATCH --time=11:30:00
#SBATCH --output tune-log-%J.out
#SBATCH --qos=gpu_free

module purge
module load gcc

source /home/kellner/venvs/COSMO/bin/activate
 
ipnport=$(shuf -i8000-9999 -n1)
 
python -u gridsearch_mine.py 
