#!/bin/bash

#SBATCH --get-user-env
#SBATCH --job-name=RBF-H
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=72:00:00
#SBATCH --partition=Slim
#SBATCH --mem=120000
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

#########################
# Standard python 3.8.5 #
#########################


# First we run one time python witout any module. You will see it uses OpenBLAS when the module is not loaded
export PYTHONPATH=/home/kellner/packages

source /home/kellner/.bashrc
echo "Hello"
which python
python -u gridsearch_mine.py 
