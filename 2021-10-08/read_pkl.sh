#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=load_pkl
#SBATCH --output=/fastscratch/c-panz/2021-10-08/log/read_pkl.log
#SBATCH --err=/fastscratch/c-panz/2021-10-08/log/read_pkl.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=25
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=72:00:00
#SBATCH --mem=150G
###SLURM HEADER
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanomodel

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-08
python $script_dir/read_pkl.py 
