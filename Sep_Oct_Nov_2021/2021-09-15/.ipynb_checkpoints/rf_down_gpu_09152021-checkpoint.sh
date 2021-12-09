#!/usr/bin/bash
#SBATCH --job-name=rf_down_gpu
#SBATCH --output=/fastscratch/c-panz/2021-09-15/log/rf_down_gpu.out  # %A: job ID 
#SBATCH --error=/fastscratch/c-panz/2021-09-15/log/rf_down_gpu.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=100G        # memory per cpu-core (4G is default)
#SBATCH -q dev
#SBATCH --time=08:00:00       # total run time limit (HH:MM:SS)
###SLURM HEADER
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanotest

#script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/$(date +%Y-%m-%d)
script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-09-15
result_dir=/fastscratch/c-panz/2021-09-15
python $script_dir/rf_down_09152021.py > $result_dir/rf_down_09152021.txt