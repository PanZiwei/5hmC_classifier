#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=sep_T4sup
#SBATCH --output=/fastscratch/c-panz/2021-10-07/log/chrom_sep_sup_T4.log
#SBATCH --err=/fastscratch/c-panz/2021-10-07/log/chrom_sep_sup_T4.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=25
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=10G
###SLURM HEADER
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanomodel

model=sup
version=5.0.14
sample=T4LambdaTF1

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-07
fast5_dir=/fastscratch/c-panz/2021-10-07/guppy$version.$model.tombo/$sample

python $script_dir/PerChromSeperation_sup.py --input_path $fast5_dir --output_path $script_dir
echo "###Per chromosome separation DONE"