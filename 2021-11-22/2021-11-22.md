1. Extract features using customized deepsignal package:
sbatch 1_deepsignal_custom_extract_5C.sh
sbatch 1_deepsignal_custom_extract_5mC.sh
sbatch 1_deepsignal_custom_extract_5hmC.sh

The feature lines:
2964 T4_5hmC.tsv
7080509 lambda_5C.tsv
11479540 lambda_5mC.tsv


(base) [c-panz@winter-log1 2021-11-22]$ sbatch 3_model_train.sh
Submitted batch job 219854
