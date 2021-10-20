#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=sep_T4
#SBATCH --output=/fastscratch/c-panz/2021-10-05/log/chrom_sep_supmodel_T4.log
#SBATCH --err=/fastscratch/c-panz/2021-10-05/log/chrom_sep_supmodel_T4.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=25
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=5G
###SLURM HEADER
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanomodel

sample=T4LambdaTF1
script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-05
fast5_dir=/fastscratch/c-panz/2021-09-30/guppy5.0.14_supmodel_tombo/$sample
output_dir=$fast5_dir.PerChrom
mkdir $output_dir

python $script_dir/PerChromSeperation.py --input_path $fast5_dir --output_path $output_dir
echo "###Per chromosome separation DONE"