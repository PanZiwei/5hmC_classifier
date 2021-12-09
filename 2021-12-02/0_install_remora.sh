#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=remora
#SBATCH --output=/fastscratch/c-panz/2021-12-01/log/remora_install.log
#SBATCH --err=/fastscratch/c-panz/2021-12-01/log/remora_install.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=2G
###SLURM HEADER
############### Install a conda environment containing customized pacakges for deepsignal
date
script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-12-01

cd /pod/2/li-lab/software
wget https://mirror.oxfordnanoportal.com/software/analysis/ont-guppy_5.0.16_linux64.tar.gz

#untar the Guppy 5.0.7
tar -xzvf ont-guppy_5.0.16_linux64.tar.gz
#Rename the folder name
mv ont-guppy ont-guppy-gpu_5.0.16
rm -rf ont-guppy_5.0.16_linux64.tar.gz

# create a environment for remora with python3.8
# version remora: v0.1.1
conda create -n remora python=3.8

# activate
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate remora

## Install remora package: https://github.com/nanoporetech/remora
pip install ont-remora

## Install megalodon v2.4.0 that can used remora
pip install megalodon

## Install pandas to load pre-trained model
pip install pandas

## Check software version
## 2021.12.2: in default remora, ont-pyguppy-client-lib==5.1.9, no package available
## Install Guppy v5.0.16 instead
pip install ont-pyguppy-client-lib==5.0.16

###Check verison
### python==3.8, remora==0.1.1, guppy==5.0.16, megalodon==2.4.0
