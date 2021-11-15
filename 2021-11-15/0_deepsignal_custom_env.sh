#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=deepsignal_custom
#SBATCH --output=/fastscratch/c-panz/2021-11-15/deepsignal_custom.log
#SBATCH --err=/fastscratch/c-panz/2021-11-15/deepsignal_custom.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -q inference
#SBATCH --time=06:00:00 # time
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=5G # memory pool for all cores
###SLURM HEADER
date
script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-11-15/

# create
conda create -n deepsignal_custom python=3.6
# activate
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate deepsignal_custom


##Install tombo package
conda install -c bioconda ont-tombo

# or install using pip
pip install tensorflow==1.13.1

# or install using pip
pip install tensorflow-gpu==1.13.1

echo "DeepSignal_custom env is installed!"