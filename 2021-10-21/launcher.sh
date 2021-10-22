#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=test
#SBATCH --output=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-21/test.log
#SBATCH --err=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-21/test.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=10
#SBATCH --nodes=2
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=120G
###SLURM HEADER
date
script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-21

source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate nanomodel

module load python36

#create a new ipython profile appended with the job id number
profile=job_${SLURM_JOB_ID}

echo "Creating profile_${profile}"
ipython profile create ${profile}

ipcontroller --ip="*" --profile=${profile} &
sleep 10

#srun: runs ipengine on each available core
srun ipengine --profile=${profile} --location=$(hostname) &
sleep 25

echo "Launching job for script $1"
python $1 -p ${profile}
