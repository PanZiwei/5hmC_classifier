#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=nanome # A single job name for the array
#SBATCH --output=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-11-15/nanome.log # %A: job ID %a:job array index
#SBATCH --err=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-11-15/nanome.err # %A: job ID %a:job array index
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=20G
###SLURM HEADER
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanome

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-11-15

mkdir /fastscratch/c-panz/nanome
cd /fastscratch/c-panz/nanome

module load singularity 
nextflow run TheJacksonLaboratory/nanome\
    -profile test,singularity
echo "Testing is done!"