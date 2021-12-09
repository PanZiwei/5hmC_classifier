1. Basecalling with Guppy 5.0.14 for T4_lambda and 5mC_lambda with super model
Guppy 5.0.14 - Released at 8th September 2021
https://community.nanoporetech.com/posts/guppy-v5-0-14-patch-releas
- A change in the q-score filter for modified base calling models.
Guppy v5.0.7 (20th May 2021)
Guppy 4.5 (Flip-Flop)	Guppy 5.0 (CRF)
Default DNA minimum q-score thresholds
fast	7	8
HAC	9	9
sup	N/A	10

Guppy v5.0.14
q-score -> 8

```shell
cd /pod/2/li-lab/Ziwei/Nanopore/daily/2021-09-30
sbatch 2_singularity_guppy5.0.14_gpu_T4lambda_supmodel.sh #203345
sbatch 2_singularity_guppy5.0.14_gpu_5mCLambda_supmodel.sh #203370
#sbatch --dependency=afterok:203345 3_seperate_chromosome_supmodel_T4lambda_gpu.sh #203696
sbatch --dependency=afterok:203370 3_seperate_chromosome_supmodel_lambda5mC_gpu.sh #203697
sbatch 3_seperate_chromosome_supmodel_T4lambda.sh #10704266
```
Attention: The best way is to extract the fast5 files only in /pass folder(e.g. /fastscratch/c-panz/2021-09-30/guppy5.0.14_supmodel_tombo/T4LambdaTF1/0/fastq_runid_10f3319c8e07b54bcc290c6d6d44b08bd3a1ee93_0_0.fastq, but right now I used all of them


```shell
sbatch delete.sh #208153
sbatch --dependency=afterok:208153 2_singularity_guppy5.0.14_gpu_T4lambda_supmodel.sh #208154
sbatch --dependency=afterok:208154 3_seperate_chromosome_supmodel_T4lambda_gpu.sh #208212

