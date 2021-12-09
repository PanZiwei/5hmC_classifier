#!/usr/bin/bash
#SBATCH --job-name=APL
#SBATCH --output=/fastscratch/c-panz/log/APL_persite_plotting.out  # %A: job ID 
#SBATCH --error=/fastscratch/c-panz/log/APL_persite_plotting.err

#SBATCH --qos=batch
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=20G        # memory per cpu-core (4G is default)
###SLURM HEADER
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate megalodon2.3.3

script_path=/pod/2/li-lab/Ziwei/Nanopore/daily/$(date +%Y-%m-%d)
input_path=/fastscratch/c-panz/megalodon
output_path=/fastscratch/c-panz/$(date +%Y-%m-%d)

python $script_path/draw_persite.py
echo "Methylation calling frequency plotting is done!"

