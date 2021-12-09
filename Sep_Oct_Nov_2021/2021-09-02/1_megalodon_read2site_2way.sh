#!/usr/bin/bash
#SBATCH --job-name=APL
#SBATCH --output=/fastscratch/c-panz/log/megalodon_read2site.out  # %A: job ID 
#SBATCH --error=/fastscratch/c-panz/log/megalodon_read2site.err

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

cutoff=( 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
for i in "${cutoff[@]}"
do
    echo $i
    python $script_path/megalodon_methylcall_cutoff.py -i $input_path/T4.Megalodon.per_read.prob.bed -o $output_path/T4.Megalodon.per_site.$i.tsv -n $i
    echo "T4 is done!"
    
    python $script_path/megalodon_methylcall_cutoff.py -i $input_path/5mC_lambda.Megalodon.per_read.prob.bed -o $output_path/5mC_lambda.Megalodon.per_site.$i.tsv -n $i
    echo "5mC_lambda is done!"
    
    python $script_path/megalodon_methylcall_cutoff.py -i $input_path/lambda.Megalodon.per_read.prob.bed -o $output_path/lambda.Megalodon.per_site.$i.tsv -n $i
    echo "lambda is done!"
    
    python $script_path/megalodon_methylcall_cutoff.py -i $input_path/APL.Megalodon.per_read.prob.bed -o $output_path/APL.Megalodon.per_site.$i.tsv -n $i
    echo "APL is done!"
done
echo "Methylation calling frequency is done!"

