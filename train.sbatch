#!/bin/bash
#SBATCH --job-name="s2s training 32"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --nodelist=node[4-5,18-19]
#SBATCH --time=14-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slurmbot-HMXNmCUIQsvSyuOBKa@vision.in.tum.de
#SBATCH --output=/storage/user/dco/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/user/dco/slurm/logs/slurm-%j.out

nvidia-smi

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

srun python /home/stud/dco/Desktop/s2s-2/Controllable-Audio-Generation-from-Sonic-Imitations/train.py