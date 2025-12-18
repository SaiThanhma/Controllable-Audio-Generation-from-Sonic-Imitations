#!/bin/bash
#SBATCH --job-name="train 2 s2s"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu=2
#SBATCH --gpus-per-node=2
#SBATCH --nodelist=node[4-5,14-21]
#SBATCH --time=5-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/user/dco/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/user/dco/slurm/logs/slurm-%j.out

nvidia-smi
srun python /home/stud/dco/Desktop/s2s-2/Controllable-Audio-Generation-from-Sonic-Imitations/train.py