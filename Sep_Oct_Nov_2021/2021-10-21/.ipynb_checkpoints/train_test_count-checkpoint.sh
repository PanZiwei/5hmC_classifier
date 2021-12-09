#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=train_test
#SBATCH --output=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-21/train_test.log
#SBATCH --err=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-21/train_test.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=1:00:00
#SBATCH --mem=4G
###SLURM HEADER
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanomodel

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-21

python $script_dir/train_test_count.py > $script_dir/train_test_count.txt

