I write a script to run the model with the best parameter generated by 2021-10-08. Instead of directly loading the model, I tried to run the model alone.



(nanomodel) [c-panz@sumner-log1 2021-10-23]$ sbatch rf_SMOTE_randomCV.sh
Submitted batch job 11330220

(base) [c-panz@sumner-log1 2021-10-23]$ sbatch rf_SMOTE_GridSearch.sh
Submitted batch job 11332275


(base) [c-panz@sumner-log1 2021-10-24]$ sbatch /pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-08/rf_pipeline_randomCV.sh
Submitted batch job 11333834

(base) [c-panz@sumner-log1 2021-10-24]$ sbatch /pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-23/rf_SMOTE_randomCV_20.sh
Submitted batch job 11333831