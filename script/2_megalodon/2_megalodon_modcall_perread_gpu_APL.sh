#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=APL # A single job name for the array
#SBATCH --output=/fastscratch/c-panz/2021-09-11/log/APL_%a.log  # %A: job ID %a:job array index
#SBATCH --error=/fastscratch/c-panz/2021-09-11/log/APL_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -q inference
#SBATCH --time=06:00:00 # time
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=50G # memory pool for all cores
#SBATCH --array=1-50  #The array number can be calculated from 
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
FAST5_PATH=/pod/2/li-lab/Nanopore_compare/nanopore_fast5/APL-N50-sept
FASTA_REF=/projects/li-lab/Ziwei/Nanopore/data/reference/hg38.fa

Prefix=APL
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

