#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=5C_lambda
#SBATCH --output=/fastscratch/c-panz/2021-11-21/log/5C_lambda.log
#SBATCH --err=/fastscratch/c-panz/2021-11-21/log/5C_lambda.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=60G
###SLURM HEADER
#### The script is used to extract features with 5mC_lambda and 5C_lambda
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate deepsignal_0.1.9

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-11-21

prefix=lambda_5C
lambda_5C=/pod/2/li-lab/Ziwei/Nanopore/data/single_read/T4LambdaTF1_guppy5.0.11_supmodel_tombo.PerChrom/J02459.1
lambda_5mC=/pod/2/li-lab/Ziwei/Nanopore/data/single_read/5mCLambdaTF1_guppy5.0.11_supmodel_tombo.PerChrom/J02459.1
ref_dir=/pod/2/li-lab/Ziwei/Nanopore/data/reference
output_dir=/fastscratch/c-panz/2021-11-21

deepsignal extract --fast5_dir $lambda_5C --reference_path $ref_dir/Lambda_phage.fa --write_path $output_dir/$prefix.tsv --corrected_group RawGenomeCorrected_001 --nproc 100 --methy_label 0

echo "Feature extraction is done!"