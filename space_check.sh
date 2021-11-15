#!/usr/bin/bash
#SBATCH --job-name=space
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org
#SBATCH --output=/pod/2/li-lab/Ziwei/Nanopore/daily/space.out 
#SBATCH --error=/pod/2/li-lab/Ziwei/Nanopore/daily/space.err

#SBATCH --ntasks=5
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=14:00:00
#SBATCH --mem-per-cpu=4G
###SLURM HEADER

du -h /pod/2/li-lab --max-depth=2 | sort -hr > /pod/2/li-lab/Ziwei/space_check_20211105.txt