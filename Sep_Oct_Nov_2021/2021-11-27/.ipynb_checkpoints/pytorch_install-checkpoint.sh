#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=pytorch
#SBATCH --output=/fastscratch/c-panz/2021-11-27/pytorch.log
#SBATCH --err=/fastscratch/c-panz//2021-11-27/pytorch.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=4G
###SLURM HEADER
# create
script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-11-27

conda env remove --name pytorch
conda create --name pytorch python==3.8
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate pytorch

# install Pytorch : https://pytorch.org/get-started/locally/#linux-python
pip3 install torch torchvision torchaudio

##Install jupyterlab
pip install jupyterlab

##Install pandas, numpy, os, sys
pip install pandas
pip install numpy

