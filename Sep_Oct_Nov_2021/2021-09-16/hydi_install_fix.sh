#!/usr/bin/bash
#SBATCH --job-name=hydi
#SBATCH --output=/fastscratch/c-panz/2021-09-16/log/hydi.out  # %A: job ID 
#SBATCH --error=/fastscratch/c-panz/2021-09-16/log/hydi.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=20G
###SLURM HEADER
date
#conda env remove --name hydi
conda create -n hydi python=2.7
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate hydi
module load gcc #load gcc
conda install -c conda-forge gcc_impl_linux-64
conda install -c anaconda make #install make
conda install -c anaconda git #install git
conda install -c conda-forge gsl #install gsl
conda install -c anaconda zlib #install zlib
conda install -c nogil-staging gmp #install gmp

ls /projects/li-lab/Ziwei/Anaconda3/envs/hydi/lib/libgsl.so -lh


