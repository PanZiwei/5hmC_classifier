#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=train_model
#SBATCH --output=/fastscratch/c-panz/2021-11-29/log/train_model.log
#SBATCH --err=/fastscratch/c-panz/2021-11-29/log/train_model.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH --gres=gpu:4            # number of gpus per node
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=20G        # memory per cpu-core (4G is default)
#SBATCH -q training
#SBATCH --time=48:00:00       # total run time limit (HH:MM:SS)
###SLURM HEADER
#### The script is used to extract features with 5mC_lambda and 5C_lambda
date
script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-11-29
software_dir=/pod/2/li-lab/Ziwei/Nanopore/deepsignal_custom

input_path=/fastscratch/c-panz/2021-11-22/deepsignal_custom
output_path=/fastscratch/c-panz/2021-11-29

# 4. Train the model
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate deepsignal_custom_1129
### Problem: Right now I can't use GPU because there is a error saying that "ImportError: libcublas.so.10.0: cannot open shared object file: No such file or directory"
CUDA_VISIBLE_DEVICES=0 deepsignal train --train_file $output_path/feature.train.tsv --valid_file $output_path/feature.valid.tsv --model_dir $output_path  --log_dir $output_path --class_num 3
echo "Model training is done!"
