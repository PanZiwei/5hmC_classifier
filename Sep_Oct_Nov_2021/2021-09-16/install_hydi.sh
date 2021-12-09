#!/bin/bash
#SBATCH --job-name=hydi_install
#SBATCH -q batch
#SBATCH -N 1 # number of nodes
#SBATCH -n 2 # number of cores
#SBATCH --mem 30g # memory pool for all cores
#SBATCH -t 72:00:00 # time (D-HH:MM)
#SBATCH -o %x.%j.out # STDOUT
#SBATCH -e %x.%j.err # STDERR

conda create -n hydi2 python=2.7 #python 2.7 is in need for ‘vcfs2tab.py’
source ~/anaconda3/etc/profile.d/conda.sh
conda activate hydi2
conda env list
module load gcc
conda install -c conda-forge gcc_impl_linux-64
conda install -c ehmoussi make
conda install -c conda-forge git
conda install -c conda-forge gsl==2.6
conda install -c conda-forge zlib
conda install -c nogil-staging gmp

