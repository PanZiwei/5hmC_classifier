#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=cpu_remora
#SBATCH --output=/fastscratch/c-panz/2021-12-03/log/remora_cpu_run.log
#SBATCH --err=/fastscratch/c-panz/2021-12-03/log/remora_cpu_run.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=50G
###SLURM HEADER
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate remora
script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-12-03

### Get Guppy cpu version since remora is relatively small and cpu is recommended
# cd /pod/2/li-lab/software
# wget https://mirror.oxfordnanoportal.com/software/analysis/ont-guppy-cpu_5.0.16_linux64.tar.gz
# tar -xzvf  ont-guppy-cpu_5.0.16_linux64.tar.gz
# mv ont-guppy-cpu ont-guppy-cpu_5.0.16
# rm -rf ont-guppy-cpu_5.0.16_linux64.tar.gz

GUPPY_PATH=/pod/2/li-lab/software/ont-guppy-cpu_5.0.16/bin
FAST5_PATH=/pod/2/li-lab/Ziwei/Nanopore/daily/test/T4LambdaTF1_test
REF_PATH=/pod/2/li-lab/Ziwei/Nanopore/data/reference
OUTPUT_PATH=/fastscratch/c-panz/2021-12-03/remora_cpu

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
    --processes 200
    

### In the older version the Megalodon will cost ~2h to finish, but remora didn't finish after 6h