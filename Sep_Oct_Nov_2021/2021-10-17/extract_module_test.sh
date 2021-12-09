#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=extract_module
#SBATCH --output=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-17/extract_module_test.log
#SBATCH --err=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-17/extract_module_test.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=120G
###SLURM HEADER
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanomodel

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-17
cd $script_dir

fast5_path='/pod/2/li-lab/Ziwei/Nanopore/daily/test/test'
ref_path='/pod/2/li-lab/Ziwei/Nanopore/data/reference/hg38.fa'
output_file='/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-17/extract_module_test.csv'

python $script_dir/cytosine/extract_module.py --fast5_path $fast5_path --ref_path $ref_path --output_path $output_file --mod_label 1
echo "###Testing is done!"

