#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=deepsignal_custom
#SBATCH --output=/fastscratch/c-panz/2021-11-26/log/deepsignal_custom.log
#SBATCH --err=/fastscratch/c-panz/2021-11-26/log/deepsignal_custom.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=100G        # memory per cpu-core (4G is default)
#SBATCH -q training
#SBATCH --time=240:00:00       # total run time limit (HH:MM:SS)
###SLURM HEADER
#### The script is used to extract features with 5mC_lambda and 5C_lambda
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate deepsignal_custom

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-11-26
software_dir=/pod/2/li-lab/Ziwei/Nanopore/deepsignal_custom

input_path=/fastscratch/c-panz/2021-11-22/deepsignal_custom
output_path=/fastscratch/c-panz/2021-11-26

###lambda_5C.tsv : 7080509  lines
###lambda_5mC.tsv: 11479540 lines
###T4_5hmC.tsv: 2964 lines

####Split the T4_5hmC
python $software_dir/scripts/randsel_file_rows.py --ori_filepath $input_path/T4_5hmC.tsv --write_filepath $output_path/T4_5hmC.random.tsv --num_lines 2964 --header false 
head -2668 $output_path/T4_5hmC.random.tsv > $output_path/T4_5hmC.train.tsv
tail -296 $output_path/T4_5hmC.random.tsv > $output_path/T4_5hmC.valid.tsv
for i in {1..20};do cat $output_path/T4_5hmC.train.tsv >> $output_path/T4_5hmC.train.repeated.tsv; done
echo "T4 mimic is done!"

# 2. randomly select equally number of 5mC/5C samples ->  split samples for training/validating(9:1)
#### Use minimum requirement: 512*100 samples = 51200 
num_feature=63424 # 59280+296*14 (5hmC:5mC:5C = 1:14:14)
python $software_dir/scripts/randsel_file_rows.py --ori_filepath $input_path/lambda_5mC.tsv --write_filepath $output_path/lambda_5mC.tsv --num_lines $num_feature --header false 
python $software_dir/scripts/randsel_file_rows.py --ori_filepath $input_path/lambda_5C.tsv --write_filepath $output_path/lambda_5C.tsv --num_lines $num_feature --header false 

# 4. split samples for training/validating(9:1) (training: 59280, testing: 113835)
head -59280 $output_path/lambda_5mC.tsv > $output_path/lambda_5mC.train.tsv
tail -4144 $output_path/lambda_5mC.tsv > $output_path/lambda_5mC.valid.tsv
head -59280 $output_path/lambda_5C.tsv > $output_path/lambda_5C.train.tsv
tail -4144 $output_path/lambda_5C.tsv > $output_path/lambda_5C.valid.tsv
echo "Lambda is done!"

# 3. combine lambda_5mC/lambda_5C/T4_5hmC for training
python $software_dir/scripts/concat_two_files.py --fp1 $output_path/lambda_5mC.train.tsv --fp2 $output_path/lambda_5C.train.tsv --concated_fp $output_path/lambda.train.tsv
python $software_dir/scripts/concat_two_files.py --fp1 $output_path/lambda.train.tsv --fp2 $output_path/T4_5hmC.train.repeated.tsv --concated_fp $output_path/feature.train.tsv

# 3. combine lambda_5mC/lambda_5C/T4_5hmC for testing
python $software_dir/scripts/concat_two_files.py --fp1 $output_path/lambda_5mC.valid.tsv --fp2 $output_path/lambda_5C.valid.tsv --concated_fp $output_path/lambda.valid.tsv
python $software_dir/scripts/concat_two_files.py --fp1 $output_path/lambda.valid.tsv --fp2 $output_path/T4_5hmC.valid.tsv --concated_fp $output_path/feature.valid.tsv
echo "Combine is done!"

cd $output_path
rm -rf T4_5hmC.valid.tsv T4_5hmC.train.tsv T4_5hmC.train.repeated.tsv T4_5hmC.random.tsv
rm -rf lambda.valid.tsv lambda.train.tsv
rm -rf lambda_5mC.train.tsv lambda_5C.train.tsv
rm -rf lambda_5mC.valid.tsv lambda_5C.valid.tsv

# 4. Train the model
CUDA_VISIBLE_DEVICES=0 deepsignal train --train_file $output_path/feature.train.tsv --valid_file $output_path/feature.valid.tsv --model_dir $output_path --class_num 3 --display_step 116
echo "Model training is done!"
