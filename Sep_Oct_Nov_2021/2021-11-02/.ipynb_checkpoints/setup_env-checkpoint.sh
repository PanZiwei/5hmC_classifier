#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=setup # A single job name for the array
#SBATCH --error=/fastscratch/c-panz/2021-11-02/log/setup.err # %A: job ID %a:job array index
#SBATCH --output=/fastscratch/c-panz/2021-11-02/log/setup.log # %A: job ID %a:job array index
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=20G
###SLURM HEADER
date
script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-11-02

##Delete previous environment
conda remove --name nanomodel_python3.8 --all

###Create a new one
conda create -n nanomodel_python3.8 python=3.8
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanomodel_python3.8


pip install pandas
pip install numpy

##Install ont-fast5-api
pip install ont-fast5-api

pip install jupyterlab

pip install -U scikit-learn
pip install imbalanced-learn

pip install statsmodels

# Install data visualization tool
pip install -U matplotlib
pip install seaborn

# Install tensorflow2
pip install tensorflow

