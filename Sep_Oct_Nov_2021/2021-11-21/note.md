# Use DeepSignal default model to train a binary model for 5mC_lambda/5C_lambda
1. Extract features from 5mC_lambda/5C_lambda with original DeepSignal script
Feature files:
lambda_5mC: `/fastscratch/c-panz/2021-11-21/lambda_5mC.tsv`
lambda_5C: `/fastscratch/c-panz/2021-11-21/lambda_5C.tsv`
```shell
(deepsignal_0.1.9) [c-panz@sumner-log2 2021-11-21]$ sbatch 1_extract_lambda_5C.sh
Submitted batch job 11781565
(deepsignal_0.1.9) [c-panz@sumner-log2 2021-11-21]$ sbatch 1_extract_lambda_5mC.sh
Submitted batch job 11781566
```

The DeepSignal has a minimum feature number for samples: 512*2000 samples
https://github.com/bioinfomaticsCSU/deepsignal/issues/74#issuecomment-975197726
For the accuracy is 0 issue, the training process checks the model paramters after trainng every batch_size*display_step samples. So I guess the training data you used has less than 512*2000 samples, which makes the training process didn't check the current best model paramters. Trying a smaller display_step and/or a smaller batch_size may make it work.

Feature counts:
2964 T4_5hmC.tsv
7080509 lambda_5C.tsv
11479540 lambda_5mC.tsv

(The lambda_5C and lambda_5mC should be the same as the Deepsignl_custom in 2021.11.22)

2. Extract formal features
`/fastscratch/c-panz/2021-11-21/formal/`
Total: lambda.feature.tsv (training: testing = 9:1, lambda.feature.train.tsv, lambda.feature.valid.tsv)
Total: lambda.feature.bin (training: testing = 9:1, lambda.feature.train.bin, lambda.feature.valid.bin)

3. Traing the model with full features
sbatch --dependency=afterok:219684 3_model_build.sh
(base) [c-panz@winter-log1 2021-11-21]$ sbatch --dependency=afterok:219684 3_model_build.sh
Submitted batch job 219685

4. Train the model with least feature numbers (test.sh)