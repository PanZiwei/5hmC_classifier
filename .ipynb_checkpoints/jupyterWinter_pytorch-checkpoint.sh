#!/bin/bash
# set the number of nodes
#SBATCH --job-name=Jup_winter
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziwei.pan@jax.org
#SBATCH --output=/pod/2/li-lab/Ziwei/Nanopore/winter_pytorch.out 
#SBATCH --error=/pod/2/li-lab/Ziwei/Nanopore/winter_pytorch.err

#SBATCH --nodes=1  # number of nodes
#SBATCH --ntasks=1 # number of cores
#SBATCH --cpus-per-task=1
#SBATCH -p gpu
#SBATCH -q dev
#SBATCH --gres gpu:1
# set max wallclock time
#SBATCH --time=08:00:00
#SBATCH --mem=50G
localcores=${SLURM_TASKS_PER_NODE}

if [ “$1” ]
then
    nenv=$1
fi

#export PATH=/projects/li-lab/Ziwei/Anaconda3/bin:$PATH

source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate pytorch

PORT=10006
HOST=$(hostname -A)

jupyter lab --ip=$HOST --port=$PORT --no-browser --notebook-dir=/projects/li-lab/Ziwei/Nanopore/daily
