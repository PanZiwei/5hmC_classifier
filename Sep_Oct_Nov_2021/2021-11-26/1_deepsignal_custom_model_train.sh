#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=deepsignal_custom
#SBATCH --output=/fastscratch/c-panz/2021-11-26/log/deepsignal_custom_train.log
#SBATCH --err=/fastscratch/c-panz/2021-11-26/log/deepsignal_custom_train.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=50G        # memory per cpu-core (4G is default)
#SBATCH -q training
#SBATCH --time=240:00:00       # total run time limit (HH:MM:SS)
###SLURM HEADER
#### The script is used to extract features with 5mC_lambda and 5C_lambda
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate deepsignal_custom

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-11-26
software_dir=/pod/2/li-lab/Ziwei/Nanopore/deepsignal_custom

input_path=/fastscratch/c-panz/2021-11-22
output_path=/fastscratch/c-panz/2021-11-26

# 4. Train the model
CUDA_VISIBLE_DEVICES=0 deepsignal train --train_file $input_path/feature.train.tsv --valid_file $input_path/feature.valid.tsv --model_dir $output_path  --log_dir $output_path --class_num 3 --display_step 116
echo "Model training is done!"
