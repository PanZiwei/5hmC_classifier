#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=rf_grid_pipeline
#SBATCH --output=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-13/rf_pipeline.log
#SBATCH --err=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-13/rf_pipeline.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=50
#SBATCH --nodes=1
#SBATCH --qos=long   #batch < 72h; long~300h
#SBATCH --time=96:00:00
#SBATCH --mem=150G
###SLURM HEADER
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanomodel

#https://jacksonlaboratory.sharepoint.com/sites/ResearchIT/SitePages/What%20are%20the%20Cluster%20SLURM%20Settings%20and%20Job%20Limits.aspx

###160 * 3 parameter in total
script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-13
#input_file=/pod/2/li-lab/Ziwei/Nanopore/daily/test/total.test.bed.gz
input_file='/pod/2/li-lab/Ziwei/Nanopore/results/megalodon2.3.4_guppy5.0.14/total.Megalodon.per_read.prob.bed.gz'

python $script_dir/rf_pipeline_gridsearch.py --input_path $input_file --output_path $script_dir > $script_dir/rf_params_smote_gridsearch.txt

