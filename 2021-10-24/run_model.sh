#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=run_model
#SBATCH --output=/fastscratch/c-panz/2021-10-24/log/run_model.log
#SBATCH --err=/fastscratch/c-panz/2021-10-24/log/run_model.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=72:00:00
#SBATCH --mem=200G
###SLURM HEADER

"""
The script is used to run the model from 2021-10-08
"""

date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanomodel

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-24
#input_file=/pod/2/li-lab/Ziwei/Nanopore/daily/test/total.test.bed.gz
input_file=/pod/2/li-lab/Ziwei/Nanopore/results/megalodon2.3.4_guppy5.0.14/total.Megalodon.per_read.prob.bed.gz

#Usage: python run_model.py --input_path $input_path --output_path $output_path
python $script_dir/run_model.py --input_path $input_file --output_path $script_dir > $script_dir/run_model.txt


