#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=5C_lambda
#SBATCH --output=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-18/lambda_5C.log
#SBATCH --err=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-18/lambda_5C.err
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

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-18
cd $script_dir

ref_path='/pod/2/li-lab/Ziwei/Nanopore/data/reference'

##lamba_5C
fast5_path='/pod/2/li-lab/Ziwei/Nanopore/data/single_read/T4LambdaTF1_guppy5.0.11_supmodel_tombo.PerChrom/J02459.1'
output_file='/pod/2/li-lab/Ziwei/Nanopore/results/feature_Guppy5.0.11/lambda_5C.csv'
python $script_dir/cytosine/extract_module.py --fast5_path $fast5_path --ref_path $ref_path/Lambda_phage.fa --output_path $output_file --mod_label 0
echo "lamba_5C is done!"


