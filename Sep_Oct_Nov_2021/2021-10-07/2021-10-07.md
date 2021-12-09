I redo the Guppy basecalling with default model to see whether I can get T4 files or not

sbatch singularity_guppy5.0.14_gpu_T4lambda_hacmodel.sh #Submitted batch job 209519
sbatch singularity_guppy5.0.14_gpu_T4lambda_supmodel.sh #Submitted batch job 209544

sbatch PerChromSeperation_hacmodel.sh 
sbatch PerChromSeperation_supmodel.sh 

#(base) [c-panz@winter-log1 2021-10-07]$ sbatch PerChromSeperation_hacmodel.sh
#Submitted batch job 211121
#(base) [c-panz@winter-log1 2021-10-07]$ sbatch PerChromSeperation_supmodel.sh
#Submitted batch job 211122



(base) [c-panz@winter-log2 2021-10-07]$ sbatch singularity_guppy5.0.14_gpu_T4lambda_supmodel.sh
Submitted batch job 212549
(base) [c-panz@winter-log2 2021-10-07]$ sbatch --dependency=afterok:212549 PerChromSeperation_supmodel_gpu.sh
Submitted batch job 212575

(base) [c-panz@winter-log2 2021-10-07]$ sbatch singularity_guppy5.0.11_gpu_T4lambda_supmodel.sh
Submitted batch job 212629
(base) [c-panz@winter-log2 2021-10-07]$ sbatch --dependency=afterok:212629 PerChromSeperation_supmodel_gpu_5.0.11.sh
Submitted batch job 212635