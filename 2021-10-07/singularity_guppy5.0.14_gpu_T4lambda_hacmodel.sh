#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=T4 # A single job name for the array
#SBATCH --error=/fastscratch/c-panz/2021-10-07/log/T4lambda_hac_%a.err # %A: job ID %a:job array index
#SBATCH --output=/fastscratch/c-panz/2021-10-07/log/T4lambda_hac_%a.log # %A: job ID %a:job array index
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -q inference
#SBATCH --time=06:00:00 # time
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=40G # memory pool for all cores
#SBATCH --array=[0-764]%200  #The array number can be calculated from 
###SLURM HEADER
date
module load singularity
module load cuda11.1/toolkit

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

model=hac
version=5.0.14
sample=T4LambdaTF1

script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-07
sif_dir=/pod/2/li-lab/Ziwei/sif
input_dir=/projects/li-lab/Ziwei/Nanopore/data/single_read/5mCLambdaTF1_200
fast5_dir=/fastscratch/c-panz/2021-10-07/guppy$version.$model.tombo/$sample
##Pay attention to the reference genome!!
refGenome=/projects/li-lab/Ziwei/Nanopore/data/reference/hg38_T4_147_lambda.fa

# Specify where our Singularity image is located
export SINGULARITY_CACHEDIR="/projects/li-lab/Ziwei/singularity-cache"

processors=72
correctedGroup="RawGenomeCorrected_001"
basecallGroup="Basecall_1D_001"

### Run guppy on slurm with default high accuracy(HAC) model
### dna_r9.4.1_450bps_hac.cfg
cd $sif_dir
singularity exec --nv guppy-gpu_$version.sif guppy_basecaller --input_path $input_dir/$((SLURM_ARRAY_TASK_ID)) --save_path $fast5_dir/$((SLURM_ARRAY_TASK_ID)) --config dna_r9.4.1_450bps_hac.cfg --gpu_runners_per_device $processors --chunks_per_runner 128 --fast5_out --verbose_logs --device cuda:0
echo "###   Basecalling DONE"

source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanotest
## Re-squiggle the data:
tombo resquiggle $fast5_dir/$((SLURM_ARRAY_TASK_ID))/workspace $refGenome --processes $processors --corrected-group $correctedGroup --basecall-group $basecallGroup --overwrite --threads-per-process $processors --include-event-stdev
echo "###   Re-squiggling DONE"





