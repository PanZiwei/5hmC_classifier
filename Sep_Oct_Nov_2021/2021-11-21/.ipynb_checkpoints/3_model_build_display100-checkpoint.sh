#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=display100
#SBATCH --output=/fastscratch/c-panz/2021-11-21/log/deepsignal_training_display100.log
#SBATCH --err=/fastscratch/c-panz/2021-11-21/log/deepsignal_training_display100.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH --gres=gpu:4            # number of gpus per node
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=100G        # memory per cpu-core (4G is default)
#SBATCH -q training
#SBATCH --time=72:00:00       # total run time limit (HH:MM:SS)
###SLURM HEADER
#### The script is used to extract features with 5mC_lambda and 5C_lambda
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate deepsignal_0.1.9

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-11-21

software_dir=/pod/2/li-lab/Ziwei/Nanopore/deepsignal_custom
input_path=/fastscratch/c-panz/2021-11-21/formal
output_path=/fastscratch/c-panz/2021-11-21/display100

# 5. train (Use bin file to accelerate the process)
CUDA_VISIBLE_DEVICES=0 deepsignal train --train_file $input_path/lambda.feature.train.bin --valid_file $input_path/lambda.feature.valid.bin --is_binary yes --model_dir $output_path --log_dir $output_path
echo "Model training is done!"