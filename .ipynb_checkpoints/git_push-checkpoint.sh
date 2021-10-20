#!/usr/bin/bash
#SBATCH --job-name=git_push
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org
#SBATCH --output=/pod/2/li-lab/Ziwei/Nanopore/git_push.out 
#SBATCH --error=/pod/2/li-lab/Ziwei/Nanopore/git_push.err

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=16:00:00
#SBATCH --mem-per-cpu=5G
###SLURM HEADER

cd /pod/2/li-lab/Ziwei/Nanopore/daily
git add --all
git commit -m "test git_push automatation & udpate gitignore"
git push

