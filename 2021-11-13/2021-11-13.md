cd /pod/2/li-lab/Ziwei/Nanopore/daily/2021-11-13
sbatch rf_ratio_GridSearch.sh
sbatch rf_ratio_RandomSearch.sh

(base) [c-panz@sumner-log1 2021-11-13]$ sbatch rf_ratio_GridSearch.sh
Submitted batch job 11705607
(base) [c-panz@sumner-log1 2021-11-13]$ sbatch rf_ratio_RandomSearch.sh
Submitted batch job 11705608

Continue to get the rror for RandomSearch with the largest portion:
```shell
joblib.externals.loky.process_executor.TerminatedWorkerError: A worker process managed by the executor was unexpectedly terminated. This could be caused by a segmentation fault while calling the function or by an excessive memory usage causing the Operating System to kill the worker.

The exit codes of the workers are {SIGKILL(-9)}
slurmstepd: error: Detected 2284 oom-kill event(s) in step 11705445.batch cgroup. Some of your processes may have been killed by the cgroup out-of-memory handler.
```

Strategy: I am using high memory node to test follwing https://jacksonlaboratory.sharepoint.com/sites/ResearchIT/SitePages/What%20are%20the%20Cluster%20SLURM%20Settings%20and%20Job%20Limits.aspx