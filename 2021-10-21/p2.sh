#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=p2_ENN
#SBATCH --output=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-21/p2_ENN.log
#SBATCH --err=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-21/p2_ENN.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=50
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=72:00:00
#SBATCH --mem=750G
###SLURM HEADER
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanomodel

#https://jacksonlaboratory.sharepoint.com/sites/ResearchIT/SitePages/What%20are%20the%20Cluster%20SLURM%20Settings%20and%20Job%20Limits.aspx

###160 * 3 parameter in total
script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-21
#input_file=/pod/2/li-lab/Ziwei/Nanopore/daily/test/total.test.bed.gz
input_file='/pod/2/li-lab/Ziwei/Nanopore/results/megalodon2.3.4_guppy5.0.14/total.Megalodon.per_read.prob.bed.gz'

python $script_dir/p2_rf_ENN_SMOTE_RandomCV.py --input_path $input_file --output_path $script_dir/result > $script_dir/p2.txt

