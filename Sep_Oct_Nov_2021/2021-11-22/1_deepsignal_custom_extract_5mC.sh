#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=5mC_deepsignal
#SBATCH --output=/fastscratch/c-panz/2021-11-22/log/deepsignal_custom_extract_5mC.log
#SBATCH --err=/fastscratch/c-panz/2021-11-22/log/deepsignal_customextract_5mC.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=60G
###SLURM HEADER
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate deepsignal_custom

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-11-22

prefix=lambda_5mC
path_5hmC=/pod/2/li-lab/Ziwei/Nanopore/data/single_read/5mCLambdaTF1_guppy5.0.11_supmodel_tombo.PerChrom/KJ477685.1
path_5mC=/pod/2/li-lab/Ziwei/Nanopore/data/single_read/5mCLambdaTF1_guppy5.0.11_supmodel_tombo.PerChrom/J02459.1
path_5C=/pod/2/li-lab/Ziwei/Nanopore/data/single_read/T4LambdaTF1_guppy5.0.11_supmodel_tombo.PerChrom/J02459.1

ref_dir=/pod/2/li-lab/Ziwei/Nanopore/data/reference
output_dir=/fastscratch/c-panz/2021-11-22/deepsignal_custom

deepsignal extract --fast5_dir $path_5mC --reference_path $ref_dir/Lambda_phage.fa --write_path $output_dir/$prefix.tsv --corrected_group RawGenomeCorrected_001 --nproc 100 --methy_label 1
echo "5mC feature extraction is done!"

