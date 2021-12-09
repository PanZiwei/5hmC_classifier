#!/usr/bin/bash
#SBATCH --job-name=rf
#SBATCH --output=/fastscratch/c-panz/2021-09-13/log/rf.out  # %A: job ID 
#SBATCH --error=/fastscratch/c-panz/2021-09-13/log/rf.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=30G
###SLURM HEADER
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanotest

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/$(date +%Y-%m-%d)
result_dir=/fastscratch/c-panz/$(date +%Y-%m-%d)
python -u $script_dir/rf.py > $result_dir/rf.txt