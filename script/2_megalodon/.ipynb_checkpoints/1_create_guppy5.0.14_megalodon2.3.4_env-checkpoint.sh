#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=5mC # A single job name for the array
#SBATCH --output=/fastscratch/c-panz/logs/guppy.tombo/5mClambda_%A_%a.log # %A: job ID %a:job array index
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=4G
###SLURM HEADER

#Install guppy
#singularity version
#.tar.gz verison

cd /pod/2/li-lab/software
wget https://mirror.oxfordnanoportal.com/software/analysis/ont-guppy-gpu_5.0.14_linux64.tar.gz
#untar the Guppy 5.0.7
tar -xzvf ont-guppy_5.0.14_linux64.tar.gz
#Rename the folder name
mv ont-guppy ont-guppy-gpu_5.0.14

#Create megalodon envinronment
conda create -n megalodon2.3.4 python=3.8
conda activate megalodon2.3.4
pip install megalodon==2.3.4

conda install -c bioconda samtools
conda list samtools

#install taiyaki
cd /home/c-panz/software/taiyaki
python3 setup.py install
#(megalodon2.3.4) [c-panz@sumner-log1 taiyaki]$ conda list taiyaki
# packages in environment at /projects/li-lab/Ziwei/Anaconda3/envs/megalodon2.3.4:
#
# Name                    Version                   Build  Channel
taiyaki                   5.3.0                    pypi_0    pypi
#install gnu-parallel: https://anaconda.org/conda-forge/parallel
conda install -c conda-forge parallel