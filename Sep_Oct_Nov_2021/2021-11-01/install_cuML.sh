#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=cuML
#SBATCH --output=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-11-01/cuML.log
#SBATCH --err=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-11-01/cuML.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=72:00:00
#SBATCH --mem=50G
###SLURM HEADER
date
##Install cuML via Conda: https://rapids.ai/start.html#rapids-release-selector
conda create -n rapids-21.10 -c rapidsai -c nvidia -c conda-forge \
    rapids-blazing=21.10 python=3.7 cudatoolkit=11.0

