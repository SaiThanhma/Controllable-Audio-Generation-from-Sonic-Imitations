#!/bin/bash
#SBATCH --job-name="train 1 s2s"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --nodelist=node[4-5,14-21]
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --time=5-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
srun python /home/stud/dco/Desktop/s2s-2/Controllable-Audio-Generation-from-Sonic-Imitations/train.py

# Comment (Lukas K.)
# For a normal DL task, it is useful to use multiple CPUs per GPU for example --cpus-per-task=3. This depends on how many CPUs your dataloader (pytorch num_workers) can use effectively.
# For a normal DL task, one should use about 32G of memory. Of course, this depends on the task. Try not to take too much, but if the memory is insufficient the task will get killed.