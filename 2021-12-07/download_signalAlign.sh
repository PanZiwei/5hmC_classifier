#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=download
#SBATCH --output=/fastscratch/c-panz/2021-12-07/log/download_signalAlign.log
#SBATCH --err=/fastscratch/c-panz/2021-12-07/log/download_signalAlign.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=72:00:00
#SBATCH --mem=5GB
###SLURM HEADER

## Download SignalAlign github
cd /fastscratch/c-panz/2021-12-07/
git clone https://github.com/ArtRand/CytosineMethylationAnalysis.git
