#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=external_test
#SBATCH --output=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-28/external_test.log
#SBATCH --err=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-28/external_test.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=72:00:00
#SBATCH --mem=100G
###SLURM HEADER
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanomodel

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-28
input_path=/pod/2/li-lab/Ziwei/Nanopore/results/megalodon2.3.4_guppy5.0.14/total.Megalodon.per_read.prob.bed.gz

#Usage: python external_test.py --input_path $input_path --model_path $model_path --output_path $output_path

####2021-10-26
model_path=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-26/result/p1_model.pkl
output_path=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-26
python $script_dir/external_test.py --input_path $input_path --model_path $model_path --output_path $output_path/result > $output_path/rerun_test.txt
echo "2021-10-26 is done!"

#####Load the model from rf_best_20211008 for confusion matrix generation
model_path=/pod/2/li-lab/Ziwei/Nanopore/daily/result/rf_best_20211008.pkl
output_path=/pod/2/li-lab/Ziwei/Nanopore/result
python $script_dir/external_test.py --input_path $input_path --model_path $model_path --output_path $output_path > $script_dir/rerun_20211008.txt
echo "2021-10-08 is done!"
