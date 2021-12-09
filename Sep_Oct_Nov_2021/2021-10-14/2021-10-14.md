I test the DeepSignal extract module to have a test on its speed and output

sbatch 0_deepsignal_env.sh #Submitted batch job #11028510

sbatch --dependency=afterok:11028510 1_deepsignal_extract_feature.sh #Submitted batch job 11028511