#!/bin/bash
#SBATCH -JSocialNavtest
#SBATCH -A gts-awu36-paid
#SBATCH -N1 --gres=gpu:1 --gres-flags=enforce-binding --mem-per-gpu=12G
#SBATCH -t5:00:00
#SBATCH -qinferno
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=splion360@gmail.com
#SBATCH -oSocialNavTest_G1_MMD.out

cd ~/p-awu36-0/SN/scripts/
module load pytorch

srun python train.py --wandb 


