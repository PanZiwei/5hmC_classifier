#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=umap
#SBATCH --output=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-11-01/umap.log
#SBATCH --err=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-11-01/umap.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=50
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=72:00:00
#SBATCH --mem=300G
###SLURM HEADER
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanomodel

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-11-01
input_path=/pod/2/li-lab/Ziwei/Nanopore/results/megalodon2.3.4_guppy5.0.14/total.Megalodon.per_read.prob.bed.gz
#input_path=/pod/2/li-lab/Ziwei/Nanopore/daily/test/total.test.bed.gz
output_path=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-11-01

python $script_dir/UMAP.py --input_path $input_path --output_path $output_path > $output_path/umap.txt
