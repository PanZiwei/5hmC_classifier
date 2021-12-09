#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=rf_20_pipeline
#SBATCH --output=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-11/rf_pipeline_20.log
#SBATCH --err=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-11/rf_pipeline_20.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=25
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=40:00:00
#SBATCH --mem=150G
###SLURM HEADER
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanomodel

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-11
python $script_dir/rf_pipeline_gridsearch_20.py > $script_dir/rf_params_20.txt
