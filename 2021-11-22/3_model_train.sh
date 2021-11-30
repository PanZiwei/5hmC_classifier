#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=custom_deepsignal_training
#SBATCH --output=/fastscratch/c-panz/2021-11-22/log/deepsignal_training_custom.log
#SBATCH --err=/fastscratch/c-panz/2021-11-22/log/deepsignal_training_custom.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH --gres=gpu:4            # number of gpus per node
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=150G        # memory per cpu-core (4G is default)
#SBATCH -q training
#SBATCH --time=240:00:00       # total run time limit (HH:MM:SS)
###SLURM HEADER
#### The script is used to extract features with 5mC_lambda and 5C_lambda
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate deepsignal_0.1.9

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-11-22

software_dir=/pod/2/li-lab/Ziwei/Nanopore/deepsignal_custom
output_path=/fastscratch/c-panz/2021-11-22

# 5. train 
CUDA_VISIBLE_DEVICES=0 deepsignal train --train_file $output_path/feature.train.tsv --valid_file $output_path/feature.valid.tsv --model_dir $output_path --display_step 1000 --class_num 3
echo "Model training is done!"