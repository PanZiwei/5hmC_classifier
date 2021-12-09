#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=rf_cv_RandomSearch
#SBATCH --output=/fastscratch/c-panz/2021-12-08/log/rf_cv_RandomSearch.log
#SBATCH --err=/fastscratch/c-panz/2021-12-08/log/rf_cv_RandomSearch.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=130
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --partition high_mem  ##Use high memory
#SBATCH --time=72:00:00
#SBATCH --mem=3000GB
###SLURM HEADER
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanomodel_python3.8

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-12-08
#input_file=/pod/2/li-lab/Ziwei/Nanopore/daily/test/total.test.bed.gz
input_file='/pod/2/li-lab/Ziwei/Nanopore/results/megalodon2.3.4_guppy5.0.14/total.Megalodon.per_read.prob.bed.gz'

python $script_dir/rf_RandomSearch_212_cv.py --input_path $input_file --output_path $script_dir/result > $script_dir/rf_RandomSearch_212_cv.txt
