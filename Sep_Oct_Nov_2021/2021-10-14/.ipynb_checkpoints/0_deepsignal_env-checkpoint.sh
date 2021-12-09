#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=deepsignal
#SBATCH --output=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-14/deepsignalenv.log
#SBATCH --err=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-14/deepsignalenv.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=5G
###SLURM HEADER
# create
#conda create -n deepsignalenv python=3.6
# activate
script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-14

source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate deepsignalenv

# install deepsignal
pip install deepsignal

##Install tombo package
conda install -c bioconda ont-tombo

# or install using pip
pip install tensorflow==1.13.1

echo "DeepSignal env is installed!"