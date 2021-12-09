rf_downsampling_RandomSearch: 15 folds for each of 100 candidates
rf_downsampling_SMOTE_RandomSearch: 15 folds for each of 100 candidates with SMOTE
rf_downsampling_SMOTE_GridSearch: All parameter with SMOTE

```shell
sbatch rf_downsampling_SMOTE_GridSearch.sh
sbatch rf_downsampling_SMOTE_RandomSearch.sh
sbatch rf_downsampling_RandomSearch.sh
sbatch megalodon_baseline.sh

(base) [c-panz@sumner-log2 2021-11-07]$ sbatch rf_downsampling_SMOTE_GridSearch.sh
Submitted batch job 11648390
(base) [c-panz@sumner-log2 2021-11-07]$ sbatch rf_downsampling_SMOTE_RandomSearch.sh
Submitted batch job 11648391
(base) [c-panz@sumner-log2 2021-11-07]$ sbatch rf_downsampling_RandomSearch.sh
Submitted batch job 11648392
(base) [c-panz@sumner-log2 2021-11-07]$ sbatch megalodon_baseline.sh
Submitted batch job 11648393
```