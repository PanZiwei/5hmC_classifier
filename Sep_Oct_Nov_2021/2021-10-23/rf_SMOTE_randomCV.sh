#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=rf_SMOTE_RandomCV
#SBATCH --output=/fastscratch/c-panz/2021-10-23/log/rf_SMOTE_RandomCV.log
#SBATCH --err=/fastscratch/c-panz/2021-10-23/log/rf_SMOTE_RandomCV.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=25
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=72:00:00
#SBATCH --mem=700G
###SLURM HEADER
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanomodel

###160 * 3 parameter in total
script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-23
#input_file=/pod/2/li-lab/Ziwei/Nanopore/daily/test/total.test.bed.gz
input_file='/pod/2/li-lab/Ziwei/Nanopore/results/megalodon2.3.4_guppy5.0.14/total.Megalodon.per_read.prob.bed.gz'

mkdir $script_dir/result
python $script_dir/rf_SMOTE_randomCV.py --input_path $input_file --output_path $script_dir/result > $script_dir/rf_SMOTE_randomCV.txt

