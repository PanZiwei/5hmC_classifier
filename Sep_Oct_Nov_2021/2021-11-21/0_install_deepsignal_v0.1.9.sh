#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=deepsignal
#SBATCH --output=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-11-21/deepsignalenv_v0.1.9.log
#SBATCH --err=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-11-21/deepsignalenv_v0.1.9.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=5G
###SLURM HEADER
# create
#conda create -n deepsignalenv python=3.6
# activate
script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-11-21

conda env remove --name deepsignalenv
conda create --name deepsignal_0.1.9 python=3.6
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate deepsignal_0.1.9

# install deepsignal
pip install deepsignal

##Install tombo package
conda install -c bioconda ont-tombo

# or install using pip
pip install tensorflow==1.13.1
pip install tensorflow-gpu==1.13.1

######2021/11/21 Update
######I have to use tensorflow==1.12.0, Otherwise I will suffer the issue "
######More discussion:

echo "DeepSignal env is installed!"