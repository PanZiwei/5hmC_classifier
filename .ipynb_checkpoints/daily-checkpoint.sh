#!/usr/bin/bash
#SBATCH --job-name=daily
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org
#SBATCH --output=/pod/2/li-lab/Ziwei/Nanopore/daily.out 
#SBATCH --error=/pod/2/li-lab/Ziwei/Nanopore/daily.err

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=16:00:00
#SBATCH --mem-per-cpu=10G
###SLURM HEADER

mkdir /pod/2/li-lab/Ziwei/Nanopore/daily/$(date +%Y-%m-%d)
mkdir /fastscratch/c-panz/$(date +%Y-%m-%d)
mkdir /fastscratch/c-panz/$(date +%Y-%m-%d)/log
touch /pod/2/li-lab/Ziwei/Nanopore/daily/$(date +%Y-%m-%d)/$(date +%Y-%m-%d).md

localcores=${SLURM_TASKS_PER_NODE}

if [ “$1” ]
then
    nenv=$1
fi

#export PATH=/projects/li-lab/Ziwei/Anaconda3/bin:$PATH

source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanomodel_python3.8

PORT=10005
HOST=$(hostname -A)

jupyter lab --ip=$HOST --port=$PORT --no-browser --notebook-dir=/projects/li-lab/Ziwei/Nanopore/
