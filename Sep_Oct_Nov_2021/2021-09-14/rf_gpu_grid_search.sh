#!/usr/bin/bash
#SBATCH --job-name=rf_grid_search
#SBATCH --output=/fastscratch/c-panz/2021-09-14/log/rf_grid_search.out  # %A: job ID 
#SBATCH --error=/fastscratch/c-panz/2021-09-14/log/rf_grid_search.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20G        # memory per cpu-core (4G is default)
#SBATCH -q inference
#SBATCH --time=06:00:00       # total run time limit (HH:MM:SS)
###SLURM HEADER
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanotest

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/$(date +%Y-%m-%d)
result_dir=/fastscratch/c-panz/$(date +%Y-%m-%d)
python -u $script_dir/rf_grid_search.py > $result_dir/rf_grid_search.txt