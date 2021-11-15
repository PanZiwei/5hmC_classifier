#!/usr/bin/bash
#SBATCH --job-name=T4_lambda
#SBATCH --output=/fastscratch/c-panz/2021-09-11/log/merge_T4_lambda.out  # %A: job ID 
#SBATCH --error=/fastscratch/c-panz/2021-09-11/log/merge_T4_lambda.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziwei.pan@jax.org

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --qos=batch   #batch < 72h; long~300h
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=20G
###SLURM HEADER
date
source /projects/li-lab/Ziwei/Anaconda3/etc/profile.d/conda.sh
conda activate megalodon2.3.4

Prefix=T4_lambda
OUTPUT_DIR=/fastscratch/c-panz/$(date +%Y-%m-%d)/$Prefix
result_dir=/fastscratch/c-panz/$(date +%Y-%m-%d)/megalodon
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

cat $OUTPUT_DIR/*/per_read_modified_base_calls.bed | awk 'BEGIN {FS="\t"; OFS="\t"} {print $2, $4, $1, $3, $5, $6, $7}' > $OUTPUT_DIR/$Prefix.per_read_modified_base_calls.merged.bed
cd $OUTPUT_DIR
LC_ALL=C sort --parallel=24 --buffer-size=2G -k1,1 -k2,2n $Prefix.per_read_modified_base_calls.merged.bed > $Prefix.per_read_modified_base_calls.sorted.bed
echo "Sorting is done!"
rm -r $OUTPUT_DIR/*/per_read_modified_base_calls.bed
echo "Merging and sorting is done!"

##T4: KJ477685.1
##Lambda: J02459.1
awk '$1 == "KJ477685.1"' $OUTPUT_DIR/$Prefix.per_read_modified_base_calls.sorted.bed > $result_dir/T4.Megalodon.methyl_call.perRead.bed
echo "T4 extracting is done!"
awk '$1 == "J02459.1"' $OUTPUT_DIR/$Prefix.per_read_modified_base_calls.sorted.bed > $result_dir/lambda.Megalodon.methyl_call.perRead.bed
echo "Lambda extracting is done!"

#Convert prob to score
Prefix=T4
python $script_dir/script/megalodon_log2score_strand.py -i $result_dir/$Prefix.Megalodon.methyl_call.perRead.bed -o $result_dir/$Prefix.Megalodon.per_read.prob.bed
Prefix=lambda
python $script_dir/script/megalodon_log2score_strand.py -i $result_dir/$Prefix.Megalodon.methyl_call.perRead.bed -o $result_dir/$Prefix.Megalodon.per_read.prob.bed
