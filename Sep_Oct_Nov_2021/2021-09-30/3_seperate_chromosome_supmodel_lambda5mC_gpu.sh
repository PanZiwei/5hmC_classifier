#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=sep_lambda5mC
#SBATCH --output=/fastscratch/c-panz/2021-09-30/log/chrom_sep_supmodel_lambda5mC.log
#SBATCH --err=/fastscratch/c-panz/2021-09-30/log/chrom_sep_supmodel_lambda5mC.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -q inference
#SBATCH --time=06:00:00 # time
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G # memory pool for all cores
###SLURM HEADER
date
sample=5mCLambdaTF1

source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanomodel

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-09-30
fast5_dir=/fastscratch/c-panz/2021-09-30/guppy5.0.14_supmodel_tombo/$sample
output_dir=$fast5_dir.PerChrom
mkdir $output_dir

python $script_dir/guppy.tombo_PerChromSeparation.py $fast5_dir $output_dir
echo "###Per chromosome separation DONE"

#T4: KJ477685.1, lambda: J02459.1
target_dir=/pod/2/li-lab/Ziwei/Nanopore/data/single_read/$sample.guppy5.0.14.supmodel.tombo.PerChrom
mkdir $target_dir
cp -R $output_dir/J02459.1 $target_dir

output_dir=/pod/2/li-lab/Ziwei/Nanopore/data/single_read/guppy5.0.14.supmodel.tombo
mkdir $output_dir
mv $target_dir/J02459.1 $output_dir/5mC_lambda
rm -rf $target_dir
echo "Copying is done!"
