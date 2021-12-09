#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=gunzip # A single job name for the array
#SBATCH --error=/fastscratch/c-panz/2021-11-02/log/gunzip.err # %A: job ID %a:job array index
#SBATCH --output=/fastscratch/c-panz/2021-11-02/log/gunzip.log # %A: job ID %a:job array index
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=10G
###SLURM HEADER
date
cd /pod/2/li-lab/Ziwei/Nanopore/results/feature_Guppy5.0.11
gzip -c lambda_5C.csv > lambda_5C.csv.gz
gzip -c lambda_5mC.csv > lambda_5mC.csv.gz
gzip -c T4_5hmC.csv > T4_5hmC.csv.gz