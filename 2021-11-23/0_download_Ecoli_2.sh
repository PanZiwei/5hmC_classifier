#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=Ecoli_2
#SBATCH --output=/fastscratch/c-panz/2021-11-23/log/Ecoli_2.log
#SBATCH --err=/fastscratch/c-panz/2021-11-23/log/Ecoli_2.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=12G
###SLURM HEADER
# create
#conda create -n deepsignalenv python=3.6
# activate
date
script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-11-23

cd /fastscratch/c-panz/2021-11-23
mkdir Ecoli
cd Ecoli
#### Download the E.coli dataset from R9: https://www.ebi.ac.uk/ena/browser/view/PRJEB13021?show=reads
#### accession PRJEB13021 (ERR1676719 for negative control and ERR1676720 for positive control) 
#### https://github.com/comprna/METEORE/issues/21#issue-1032269641
#wget -c --tries=0 ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR167/ERR1676719/ecoli_er2925.pcr.r9.timp.061716.tar.gz
#wget -c --tries=0 ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR167/ERR1676720/ecoli_er2925.pcr_MSssI.r9.timp.061716.tar.gz
#echo "Downloading id done!"

#tar xvzf ecoli_er2925.pcr.r9.timp.061716.tar.gz
tar xvzf ecoli_er2925.pcr_MSssI.r9.timp.061716.tar.gz
echo "Untar the files!"
