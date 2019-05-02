#!/bin/sh
#SBATCH -A lbpm
#SBATCH -N 1
#SBATCH --partition=pascal
#SBATCH -t 8:00:00
#SBATCH -p pbatch
#SBATCH --export=ALL

source ~/.bashrc
source activate tfgpu

module load cuda/9.1.85
# use tensorflow-gpu-1.0.1
srun -N1 -n1 python -u run_WAE.py
