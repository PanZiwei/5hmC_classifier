#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=ENN_n_neighbor
#SBATCH --output=/fastscratch/c-panz/2021-10-22/log/ENN_n_neighbor.log
#SBATCH --err=/fastscratch/c-panz/2021-10-22/log/ENN_n_neighbor.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=50
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=72:00:00
#SBATCH --mem=700G
###SLURM HEADER
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanomodel

#https://jacksonlaboratory.sharepoint.com/sites/ResearchIT/SitePages/What%20are%20the%20Cluster%20SLURM%20Settings%20and%20Job%20Limits.aspx

###160 * 3 parameter in total
script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-22
input_file='/pod/2/li-lab/Ziwei/Nanopore/results/megalodon2.3.4_guppy5.0.14/total.Megalodon.per_read.prob.bed.gz'

python $script_dir/ENN_n_neighbor.py

