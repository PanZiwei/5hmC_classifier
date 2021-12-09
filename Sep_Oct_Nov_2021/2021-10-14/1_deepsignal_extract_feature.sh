#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=deepsignal
#SBATCH --output=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-14/deepsignal_extract.log
#SBATCH --err=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-14/deepsignal_extract.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=5G
###SLURM HEADER
# create
#conda create -n deepsignalenv python=3.6
# activate
date
script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-14

source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate deepsignalenv

fast5_path=/pod/2/li-lab/Ziwei/Nanopore/daily/test/test
ref_path=/pod/2/li-lab/Ziwei/Nanopore/data/reference/hg38.fa
output_file=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-14/deepsignal_feature.tsv


#https://github.com/bioinfomaticsCSU/deepsignal
deepsignal extract --fast5_dir $fast5_path --reference_path $ref_path --write_path $output_file --corrected_group RawGenomeCorrected_001 --nproc 10
date
echo "###DeepSignal testing is done!"

