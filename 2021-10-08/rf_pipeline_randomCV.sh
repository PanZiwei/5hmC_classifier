#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=rf_random_pipeline
#SBATCH --output=/fastscratch/c-panz/2021-10-08/log/rf_pipeline_randomCV.log
#SBATCH --err=/fastscratch/c-panz/2021-10-08/log/rf_pipeline_randomCV.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=25
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=72:00:00
#SBATCH --mem=150G
###SLURM HEADER
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanomodel

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-08
python $script_dir/rf_pipeline.py > $script_dir/rf_smote_params.txt
