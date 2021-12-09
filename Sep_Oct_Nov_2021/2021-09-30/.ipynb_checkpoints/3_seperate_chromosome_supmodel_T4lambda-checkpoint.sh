#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=sep_T4
#SBATCH --output=/fastscratch/c-panz/2021-09-30/log/chrom_sep_supmodel_T4.log
#SBATCH --err=/fastscratch/c-panz/2021-09-30/log/chrom_sep_supmodel_T4.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=25
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=5G
###SLURM HEADER
date
sample=T4LambdaTF1

source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanomodel

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-09-30
fast5_dir=/fastscratch/c-panz/2021-09-30/guppy5.0.14_supmodel_tombo/$sample
output_dir=$fast5_dir.PerChrom
mkdir $output_dir

python $script_dir/guppy.tombo_PerChromSeparation.py $fast5_dir $output_dir
echo "###Per chromosome separation DONE"

#array=($fast5_dir/*/*.txt)
#head -1 ${array[0]} > $fast5_dir/summary.txt
#tail -n +2 -q ${array[@]} >> $fast5_dir/summary.txt

#T4: KJ477685.1, lambda: J02459.1
target_dir=/pod/2/li-lab/Ziwei/Nanopore/data/single_read/$sample.guppy5.0.14.supmodel.tombo.PerChrom
mkdir $target_dir
cp -R $output_dir/KJ477685.1 $target_dir
cp -R $output_dir/J02459.1 $target_dir

output_dir=/pod/2/li-lab/Ziwei/Nanopore/data/single_read/guppy5.0.14.supmodel.tombo
mkdir $output_dir
mv -R $target_dir/KJ477685.1 $output_dir/T4
mv -R $target_dir/J02459.1 $output_dir/lambda
rm -rf $target_dir
echo "Copying is done!"
