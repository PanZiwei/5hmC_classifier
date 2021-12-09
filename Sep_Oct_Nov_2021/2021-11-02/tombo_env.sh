#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=tombo # A single job name for the array
#SBATCH --error=/fastscratch/c-panz/2021-11-02/log/tombo_setup.err # %A: job ID %a:job array index
#SBATCH --output=/fastscratch/c-panz/2021-11-02/log/tombo_setup.log # %A: job ID %a:job array index
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=10G
###SLURM HEADER
date
script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-11-02

conda create -n tombo python=3.6

source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate tombo

##Install tombo package
conda install -c bioconda ont-tombo

