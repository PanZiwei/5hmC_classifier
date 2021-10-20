# 2021/10-13

## Random forest classifier
New change:
1. I add the input_path and output_path for `rf_pipeline_gridsearch.py` to avoid overwriting
2. Increase --ntasks in the slurm header to accelerate (maximum = 70)
3. Seperate the confusion matrix code into a single file `plot_confusion_matrix.py`

Slurm header information:

https://jacksonlaboratory.sharepoint.com/sites/ResearchIT/SitePages/What%20are%20the%20Cluster%20SLURM%20Settings%20and%20Job%20Limits.aspx


Modify the confusion matrix to emphasize the percentage, instead of the CpG sites