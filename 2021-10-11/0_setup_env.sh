#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=setup # A single job name for the array
#SBATCH --error=/fastscratch/c-panz/2021-10-11/log/setup.err # %A: job ID %a:job array index
#SBATCH --output=/fastscratch/c-panz/2021-10-11/log/setup.log # %A: job ID %a:job array index
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=4G
###SLURM HEADER
#conda env remove --name nanomodel
#conda create -n nanomodel python=3.6
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanomodel

#pip install pandas
##Install tombo package
conda install -c bioconda ont-tombo
##Install ont-fast5-api
#pip install ont-fast5-api

#install jupyterlab, not necessary for formal release
#pip install jupyterlab 

#conda list -f python
#conda list -f ont-tombo ##Check ont-tombo version
#conda list -f ont-fast5-api