#!/usr/bin/bash
#SBATCH --job-name=zip
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org
#SBATCH --output=/fastscratch/c-panz/2021-09-13/log/gzip.out 
#SBATCH --error=/fastscratch/c-panz/2021-09-13/log/gzip.out 

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=5G
###SLURM HEADER
date
set -x

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-09-13
cd $script_dir
find . \( -name "*.bed" -o -name "*.tsv" -o -name "*.cov" \) -exec gzip {} \;