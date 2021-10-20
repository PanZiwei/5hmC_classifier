#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=sep_hac_T4hac
#SBATCH --output=/fastscratch/c-panz/2021-10-08/log/chrom_sep_hac_T4.log
#SBATCH --err=/fastscratch/c-panz/2021-10-08/log/chrom_sep_hac_T4.err
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

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-08
fast5_dir=/fastscratch/c-panz/2021-10-07/guppy5.0.14.hac.tombo/T4LambdaTF1
output_dir=$fast5_dir.PerChrom
mkdir $output_dir

python $script_dir/guppy.tombo_PerChromSeparation.py $fast5_dir $output_dir
echo "###Per chromosome separation DONE"

#array=($fast5_dir/*/*.txt)
#head -1 ${array[0]} > $fast5_dir/summary.txt
#tail -n +2 -q ${array[@]} >> $fast5_dir/summary.txt
