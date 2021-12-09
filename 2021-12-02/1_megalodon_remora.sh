#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=remora
#SBATCH --output=/fastscratch/c-panz/2021-12-01/log/remora_install.log
#SBATCH --err=/fastscratch/c-panz/2021-12-01/log/remora_install.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20G        # memory per cpu-core (4G is default)
#SBATCH -q inference
#SBATCH --time=06:00:00       # total run time limit (HH:MM:SS)
#SBATCH --array=1
###SLURM HEADER
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate remora
script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-12-01

GUPPY_PATH=/pod/2/li-lab/software/ont-guppy-gpu_5.0.16/bin
FAST5_PATH=/pod/2/li-lab/Ziwei/Nanopore/daily/test/T4LambdaTF1_test
REF_PATH=/pod/2/li-lab/Ziwei/Nanopore/data/reference
OUTPUT_PATH=/fastscratch/c-panz/2021-12-02/remora_result

# Guppy v5.0.14 + remora v0.1.1
# Example command to output basecalls, mappings, and CpG 5mC and 5hmC methylation in both per-read (``mod_mappings``) and aggregated (``mods``) formats
#   Compute settings: GPU devices 0 and 1 with 20 CPU cores
megalodon \
    `# input reads` \
    ${FAST5_PATH}/ \
    `# output location + overwrite` \
    --output-directory ${OUTPUT_PATH} \
    --overwrite \
    `# guppy options` \
    --guppy-server-path ${GUPPY_PATH}/guppy_basecall_server \
    --guppy-config dna_r9.4.1_450bps_hac.cfg \
    --remora-modified-bases dna_r9.4.1_e8 hac 0.0.0 5hmc_5mc CG 0 \
    `# output all the things` \
    --outputs basecalls mod_basecalls mappings \
    per_read_mods mods mod_mappings \
    per_read_refs signal_mappings \
    --reference ${REF_PATH}/hg38_T4_147_lambda.fa \
    --devices 0 \
    --processes 20
    

### In the older version the Megalodon will cost ~2h to finish, but remora didn't finish after 6h