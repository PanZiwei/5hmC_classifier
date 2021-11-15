#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=guppy # A single job name for the array
#SBATCH --error=/fastscratch/c-panz/2021-09-30/log/guppy.err # %A: job ID %a:job array index
#SBATCH --output=/fastscratch/c-panz/2021-09-30/log/guppy.err # %A: job ID %a:job array index
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=4G
###SLURM HEADER
date
module load singularity
module load cuda11.1/toolkit

###Original singularity download link: https://hub.docker.com/r/genomicpariscentre/guppy-gpu/tags

#Download singularity v5.0.14
sif_path=/pod/2/li-lab/Ziwei/sif
cd $sif_path
singularity pull docker://genomicpariscentre/guppy-gpu:5.0.14

#Download singularity v5.0.11
sif_path=/pod/2/li-lab/Ziwei/sif
cd $sif_path
singularity pull docker://genomicpariscentre/guppy-gpu:5.0.11