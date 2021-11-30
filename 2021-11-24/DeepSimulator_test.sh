#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=DeepSimulator_test
#SBATCH --output=/fastscratch/c-panz/2021-11-24/log/DeepSimulator_test.log
#SBATCH --err=/fastscratch/c-panz/2021-11-24/log/DeepSimulator_test.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=2G
###SLURM HEADER
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate tensorflow_cdpm

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-11-24

##Install DeepSimulator
software_dir=/pod/2/li-lab/Ziwei/software/DeepSimulator

cd $software_dir
./deep_simulator.sh -i example/artificial_human_chr22.fasta
