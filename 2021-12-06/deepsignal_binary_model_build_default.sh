#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=deepsignal
#SBATCH --output=/fastscratch/c-panz/2021-12-06/log/deepsignal_training.log
#SBATCH --err=/fastscratch/c-panz/2021-12-06/log/deepsignal_training.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH --gres=gpu:4            # number of gpus per node
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=100G        # memory per cpu-core (4G is default)
#SBATCH -q training
#SBATCH --time=240:00:00       # total run time limit (HH:MM:SS)
###SLURM HEADER
#### The script is used to extract features with 5mC_lambda and 5C_lambda
date
script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-12-06
software_dir=/pod/2/li-lab/Ziwei/Nanopore/deepsignal_custom
input_path=/fastscratch/c-panz/2021-11-22/deepsignal_custom
output_path=/fastscratch/c-panz/2021-12-06

source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate deepsignal_0.1.9
###lambda_5C.tsv : 7080509  lines
###lambda_5mC.tsv: 11479540 lines
# 2. randomly select equally number (e.g., 10k) of positive and negative samples (The input has to be balanced at this moment)
num_feature=7080500
python $software_dir/scripts/randsel_file_rows.py --ori_filepath $input_path/lambda_5mC.tsv --write_filepath $output_path/lambda_5mC.feature.tsv --num_lines $num_feature --header false 
python $software_dir/scripts/randsel_file_rows.py --ori_filepath $input_path/lambda_5C.tsv --write_filepath $output_path/lambda_5C.feature.tsv --num_lines $num_feature --header false 

# 3. combine positive and negative samples for training
# after combining, the combined file can be splited into two files as training/validating set, see step 4.
python $software_dir/scripts/concat_two_files.py --fp1 $output_path/lambda_5mC.feature.tsv --fp2 $output_path/lambda_5C.feature.tsv --concated_fp $output_path/lambda.feature.tsv

# 4. split samples for training/validating(9:1)
head -6372450 $output_path/lambda.feature.tsv > $output_path/lambda.feature.train.tsv
tail -708050 $output_path/lambda.feature.tsv > $output_path/lambda.feature.valid.tsv

# 4. Convert tsv into bin file
python $software_dir/scripts/generate_binary_feature_file.py --input_file $output_path/lambda.feature.train.tsv
python $software_dir/scripts/generate_binary_feature_file.py --input_file $output_path/lambda.feature.valid.tsv
echo "Feature preparation is done!"

# 5. train (Use bin file to accelerate the process)
CUDA_VISIBLE_DEVICES=0 deepsignal train --train_file $output_path/lambda.feature.train.bin --valid_file $output_path/lambda.feature.valid.bin --is_binary yes --model_dir $output_path --log_dir $output_path
echo "Model training is done!"