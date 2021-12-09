#!/usr/bin/bash
#SBATCH --job-name=rf_fix
#SBATCH --output=/fastscratch/c-panz/2021-09-29/rf_fix.out  # %A: job ID 
#SBATCH --error=/fastscratch/c-panz/2021-09-29/rf_fix.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=25
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=60:00:00
#SBATCH --mem=200G
###SLURM HEADER
date
script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-09-29
python -u $script_dir/rf_092921_100_25_10.py > $script_dir/rf_092921_100_25_10.txt