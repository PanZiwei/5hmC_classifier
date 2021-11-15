#!/usr/bin/bash
#SBATCH --job-name=APL
#SBATCH --output=/fastscratch/c-panz/2021-09-11/log/merge_APL.out  # %A: job ID 
#SBATCH --error=/fastscratch/c-panz/2021-09-11/log/merge_APL.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -q inference
#SBATCH --time=06:00:00 # time
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=20G # memory pool for all cores
###SLURM HEADER
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate megalodon2.3.4

Prefix=APL
OUTPUT_DIR=/fastscratch/c-panz/2021-09-11/$Prefix
result_dir=/fastscratch/c-panz/2021-09-11/megalodon
##Merge megalodon results: per_read_modified_base_calls.txt
##This output contains the following fields: read_id, chrm, strand, pos, mod_log_prob, can_log_prob, and mod_base
#(megalodon2.3.3) [c-panz@sumner068 0]$ head -3  per_read_modified_base_calls.txt
#read_id	chrm	strand	pos	mod_log_prob	can_log_prob	mod_base
#000b67f2-79b3-4d65-9dd0-21f3e63808eb	chr11	-	133007841	-5.141804291094167	-0.005864291660453292	m
#000b67f2-79b3-4d65-9dd0-21f3e63808eb	chr11	-	133007810	-4.171647405532821	-0.015547056536456383	m
##Remove the header of merge megalodon results, then merge the array result, and reorder the colums to begin with chromosome and start position
for dir in $OUTPUT_DIR/*; do
  if [ -d "$dir" ]; then
    cd "$dir" 
#    echo "$dir" 
    tail -n +2 per_read_modified_base_calls.txt > per_read_modified_base_calls.bed 
  fi
done
echo "Trimming is done!"

cat $OUTPUT_DIR/*/per_read_modified_base_calls.bed | awk 'BEGIN {FS="\t"; OFS="\t"} {print $2, $4, $1, $3, $5, $6, $7}' > $result_dir/$Prefix.per_read_modified_base_calls.merged.bed
rm -r $OUTPUT_DIR/*/per_read_modified_base_calls.bed
echo "Sorting is done!"

cd $result_dir
#use GNU Parallel to seperate results by main chromosomes (chr1-22,X,Y) and merge the results
parallel -j12 'cat APL.per_read_modified_base_calls.merged.bed | grep "chr{}\b" > /fastscratch/c-panz/2021-09-11/megalodon/APL.chr{}.bed' ::: {1..22} X Y
cat $result_dir/$Prefix.chr*.bed > $result_dir/$Prefix.total.bed
LC_ALL=C sort --parallel=24 --buffer-size=2G -k1,1 -k2,2n $result_dir/$Prefix.total.bed > $result_dir/$Prefix.Megalodon.methyl_call.perRead.bed
rm -rf $result_dir/APL.total.bed
rm -rf $result_dir/APL.chr*.bed

#Convert prob to score
Prefix=APL
script_dir=/pod/2/li-lab/Ziwei/Nanopore/daily/2021-09-11
python $script_dir/script/megalodon_log2score_strand.py -i $result_dir/$Prefix.Megalodon.methyl_call.perRead.bed -o $result_dir/$Prefix.Megalodon.per_read.prob.bed

