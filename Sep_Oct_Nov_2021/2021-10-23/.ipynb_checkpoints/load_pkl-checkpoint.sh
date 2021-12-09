#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=load_pkl
#SBATCH --output=/fastscratch/c-panz/2021-10-23/log/load_pkl.log
#SBATCH --err=/fastscratch/c-panz/2021-10-23/log/load_pkl.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=72:00:00
#SBATCH --mem=50G
###SLURM HEADER
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanomodel

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-23
input_file=/pod/2/li-lab/Ziwei/Nanopore/results/megalodon2.3.4_guppy5.0.14/total.Megalodon.per_read.prob.bed.gz
pkl_file=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-08/rf.pkl

#Usage: python load_pkl.py --input_path $input_path --pkl_path $pkl_path --output_path $output_path
mkdir $script_dir/result
python $script_dir/load_pkl.py --input_path $input_file --pkl_path $pkl_file --output_path $script_dir/result > $script_dir/load_20211008_pkl.txt


