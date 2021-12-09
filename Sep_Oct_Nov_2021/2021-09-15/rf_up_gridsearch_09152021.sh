#!/usr/bin/bash
#SBATCH --job-name=rf_gridsearch
#SBATCH --output=/fastscratch/c-panz/2021-09-15/log/rf_gridsearch.out  # %A: job ID 
#SBATCH --error=/fastscratch/c-panz/2021-09-15/log/rf_gridsearch.err
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
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanotest

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-09-15
result_dir=/fastscratch/c-panz/2021-09-15
python -u $script_dir/rf_gridsearch_09152021.py > $result_dir/rf_gridsearch_09152021.txt