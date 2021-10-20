#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=delete # A single job name for the array
#SBATCH --error=/fastscratch/c-panz/2021-09-30/log/delete.err # %A: job ID %a:job array index
#SBATCH --output=/fastscratch/c-panz/2021-09-30/log/delete.log # %A: job ID %a:job array index
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -q inference
#SBATCH --time=06:00:00 # time
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G # memory pool for all cores
###SLURM HEADER
date
rm -rf /fastscratch/c-panz/2021-09-30/guppy5.0.14_supmodel_tombo/T4LambdaTF1
rm -rf /fastscratch/c-panz/2021-09-30/guppy5.0.14_supmodel_tombo/T4LambdaTF1.PerChrom
mkdir /fastscratch/c-panz/2021-09-30/guppy5.0.14_supmodel_tombo/T4LambdaTF1
mkdir /fastscratch/c-panz/2021-09-30/guppy5.0.14_supmodel_tombo/T4LambdaTF1.PerChrom
echo "Deleting is done!"