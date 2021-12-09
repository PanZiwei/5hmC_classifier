#!/usr/bin/bash
#SBATCH --job-name=rf_small_randomCV
#SBATCH --output=/fastscratch/c-panz/2021-09-29/rf_randomCV.out  # %A: job ID 
#SBATCH --error=/fastscratch/c-panz/2021-09-29/rf_randomCV.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=25
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=40:00:00
#SBATCH --mem=200G
###SLURM HEADER
date
script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-09-29
python -u $script_dir/rf_randomsearch_test.py > $script_dir/rf_randomsearch_test.txt