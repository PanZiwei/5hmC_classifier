#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=extract_CG_feature # A single job name for the array
#SBATCH --error=/fastscratch/c-panz/2021-11-02/log/extract_CG_feature.err # %A: job ID %a:job array index
#SBATCH --output=/fastscratch/c-panz/2021-11-02/log/extract_CG_feature.log # %A: job ID %a:job array index
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=100G
###SLURM HEADER
date
script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-11-02
python $script_dir/extract_CG_feature.py
