#!/usr/bin/bash

###SLURM HEADER
#SBATCH --job-name=get_fast5
#SBATCH --output=/fastscratch/c-panz/2021-11-02/log/get_fast5.log
#SBATCH --err=/fastscratch/c-panz/2021-11-02/log/get_fast5.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=5
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=10G
###SLURM HEADER
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanomodel


##### The script is used to generate the fast5 list based on chromosome information after basecalling step
sample=T4LambdaTF1
script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/script/1_process_data
fast5_dir=/fastscratch/c-panz/guppy5.0.14_supmodel_tombo/$sample
output_dir=$fast5_dir.PerChrom
mkdir $output_dir

python $script_dir/list_PerChromSeparation.py --input_path $fast5_dir --output_path $output_dir
echo "###Per chromosome separation DONE"