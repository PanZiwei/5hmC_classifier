#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=copy
#SBATCH --output=/fastscratch/c-panz/2021-12-06/log/copy_feature.log
#SBATCH --err=/fastscratch/c-panz/2021-12-06/log/copy_feature.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=4G
###SLURM HEADER
#### The script is used to extract features with 5mC_lambda and 5C_lambda
date
input_path=/fastscratch/c-panz/2021-11-22/deepsignal_custom 
output_path=/pod/2/li-lab/Ziwei/Nanopore/results/deepsignal0.1.9_guppy5.0.11

#cp $input_path/lambda_5C.tsv $output_path
#cp $input_path/lambda_5mC.tsv $output_path
#cp $input_path/T4_5hmC.tsv $output_path
cd $output_path
find . \ -name "*.tsv" -exec gzip {}