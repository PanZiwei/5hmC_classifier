#!/usr/bin/bash
#SBATCH --job-name=mega_5hmC
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org
#SBATCH --output=/fastscratch/c-panz/2021-09-11/log/T4_guppy5.0.14_%a.out  # %A: job ID 
#SBATCH --error=/fastscratch/c-panz/2021-09-11/log/T4_guppy5.0.14_%a.err

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G        # memory per cpu-core (4G is default)
#SBATCH -q inference
#SBATCH --time=02:00:00       # total run time limit (HH:MM:SS)
#SBATCH --array=[0-764]%200  #The array number can be calculated from 
###SLURM HEADER
date
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate megalodon2.3.4
script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/$(date +%Y-%m-%d)

GUPPY_PATH=/pod/2/li-lab/software/ont-guppy-gpu_5.0.14/bin
SAMTOOLS_PATH=/pod/2/li-lab/Ziwei/Anaconda3/envs/megalodon2.3.4/bin/samtools
FAST5_PATH=/projects/li-lab/Ziwei/Nanopore/data/single_read/T4LambdaTF1_200
FASTA_REF=/pod/2/li-lab/Ziwei/Nanopore/data/reference/hg38_T4_147_lambda.fa

Prefix=T4_lambda
OUTPUT_DIR=/fastscratch/c-panz/$(date +%Y-%m-%d)/$Prefix

GUPPY_TIMEOUT=500
##“CPU much slower (generally about 100X slower) than when using a GPU”
#GPU version
megalodon \
    `# input reads` \
    ${FAST5_PATH}/$((SLURM_ARRAY_TASK_ID)) \
    `# output location + overwrite` \
    --output-directory ${OUTPUT_DIR}/$((SLURM_ARRAY_TASK_ID)) \
    --overwrite \
    `# output all the things` \
    --outputs basecalls mod_basecalls mappings \
    per_read_mods mods mod_mappings \
    per_read_refs signal_mappings \
    --mod-min-prob 0 \
    `# guppy options` \
    --guppy-server-path ${GUPPY_PATH}/guppy_basecall_server \
    --guppy-params "-d /pod/2/li-lab/software/rerio/basecall_models/" \
    --guppy-config res_dna_r941_min_modbases_5mC_5hmC_v001.cfg \
    --guppy-timeout ${GUPPY_TIMEOUT} \
    `# mapping settings (cram requires FASTA reference)` \
    --samtools-executable ${SAMTOOLS_PATH} \
    --sort-mappings \
    --mappings-format bam \
    --reference ${FASTA_REF} \
    `# modified base settings` \
    --mod-motif hm CG 0 \
    --mod-aggregate-method binary_threshold \
    --mod-output-formats bedmethyl wiggle \
    --write-mods-text \
    --write-mod-log-probs \
    `# number of megalodon read processing workers` \
    --devices 0
    
echo "Megalodon methylation calling is done!"

