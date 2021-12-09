#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=test_rf_RandomSearch
#SBATCH --output=/fastscratch/c-panz/2021-12-08/log/rf_RandomSearch_test.log
#SBATCH --err=/fastscratch/c-panz/2021-12-08/log/rf_RandomSearch_test.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=5
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=72:00:00
#SBATCH --mem=100GB
###SLURM HEADER
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanomodel_python3.8

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-12-08
input_file=/pod/2/li-lab/Ziwei/Nanopore/daily/test/total.test.bed.gz
#input_file='/pod/2/li-lab/Ziwei/Nanopore/results/megalodon2.3.4_guppy5.0.14/total.Megalodon.per_read.prob.bed.gz'

python $script_dir/test_rf_RandomSearch_212.py --input_path $input_file --output_path $script_dir/result > $script_dir/test_rf_RandomSearch_212.txt

