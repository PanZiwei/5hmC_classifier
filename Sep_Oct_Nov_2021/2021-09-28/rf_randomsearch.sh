#!/usr/bin/bash
#SBATCH --job-name=rf_randomCV
#SBATCH --output=/fastscratch/c-panz/2021-09-27/log/rf_randomCV.out  # %A: job ID 
#SBATCH --error=/fastscratch/c-panz/2021-09-27/log/rf_randomCV.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=70:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=80G
###SLURM HEADER
date
script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-09-28
python -u $script_dir/rf_randomsearch.py > $script_dir/rf_randomsearch.txt