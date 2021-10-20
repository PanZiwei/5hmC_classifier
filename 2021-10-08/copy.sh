#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=copy # A single job name for the array
#SBATCH --error=/fastscratch/c-panz/2021-10-08/log/cp.err # %A: job ID %a:job array index
#SBATCH --output=/fastscratch/c-panz/2021-10-08/log/cp.log # %A: job ID %a:job array index
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=60:00:00
#SBATCH --mem=200G
###SLURM HEADER
###SLURM HEADER

cd /fastscratch/c-panz
cp -R APL.fast5.gz.encrypted APL.fast5.gz.Encrypted
cp -R EGA_submission_Encrypted EGA_submission_encrypted




