#!/usr/bin/bash
###SLURM HEADER
#SBATCH --job-name=deepsignal_custom
#SBATCH --output=/fastscratch/c-panz/2021-11-22/log/deepsignal_custom_install.log
#SBATCH --err=/fastscratch/c-panz/2021-11-22/log/deepsignal_custom_install.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=2G
###SLURM HEADER
############### Install a conda environment containing customized pacakges for deepsignal
date
script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-11-22
module_dir=/pod/2/li-lab/Ziwei/Nanopore/deepsignal_custom

###Remove archive
conda env remove -n deepsignal_custom
rm -rf $module_dir/deepsignal_custom.egg-info
rm -rf $module_dir/dist

# create
conda create -n deepsignal_custom python=3.6
# activate
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate deepsignal_custom

##Install tombo package (may be not necessary)
#conda install -c bioconda ont-tombo

##Install dependencies
pip install numpy==1.16.4 #https://thoth-station.ninja/j/tf_25636.html
pip install -U scikit-learn
pip install h5py
conda install -c anaconda statsmodels

# or install tesorflow & tesnorflow_gpu
pip install tensorflow==1.13.1
pip install tensorflow-gpu==1.13.1

#### Insatll libcbals
#### Otherwise, it will get the error - ImportError: libcublas.so.10.0: cannot open shared object file: No such file or directory 
conda install libcblas==3.9.0   

## Install deepsignal_custom: There is something wrong with the installation, so I can't use the module at this moment
# install a package that includes a setup.py file
cd $module_dir
python setup.py install

###Install jupyterlab: may need to be deleted later
pip install jupyterlab

#####The deepsignal_package is installed named deepsignal-custom
#####See line 42 at setup.py