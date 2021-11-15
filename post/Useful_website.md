One hot encoding DNA: https://elferachid.medium.com/one-hot-encoding-dna-92a1c29ba15a

A tool for writing TILs (Today I Learned): https://www.danielecook.com/a-tool-for-writing-tils-today-i-learned/

Get chromosome sizes from fasta file: https://www.biostars.org/p/173963/

Python submodule imports using __init__.py: https://stackoverflow.com/questions/24302754/python-submodule-imports-using-init-py

unzip a list of tuples in Python: https://www.kite.com/python/answers/how-to-unzip-a-list-of-tuples-in-python



ML:

Develop machine learning pipeline:  https://medium.com/@cezinejang/introduction-to-machine-learning-pipelines-adb041120856
https://machinelearningmastery.com/modeling-pipeline-optimization-with-scikit-learn/

find -name "*pattern*" | wc -l

How to implement ratio-based SMOTE oversampling while CV-ing dataset: https://stackoverflow.com/a/51493479

Save the best parameter: https://stackoverflow.com/questions/34143829/sklearn-how-to-save-a-model-created-from-a-pipeline-and-gridsearchcv-using-jobli
```python
import joblib
joblib.dump(rf_random_search.best_estimator_, os.path.join(output_path,pkl_model))
```



t-SNE plotting:

t-Distributed Stochastic Neighbor Embedding (t-SNE) Hyperparameter Tuning for n_component: https://pyshark.com/visualization-of-multidimensional-datasets-using-t-sne-in-python/#t-distributed-stochastic-neighbor-embedding-in-python


Data visualization: 
1. seaborn color palette: https://seaborn.pydata.org/tutorial/color_palettes.html
2. Matlib color palette: https://matplotlib.org/stable/tutorials/colors/colormaps.html


GridSearch with predefinedsplit: https://gist.github.com/anirudhshenoy/e87525499d66e8a46a56bfa27e2d2f2f

```python
#https://stackoverflow.com/a/43766334
#https://gist.github.com/anirudhshenoy/e87525499d66e8a46a56bfa27e2d2f2f
# put -1 so they will be in training set, put 0 for others to go to test dataset
test_fold = [-1 for _ in range(X_train.shape[0])] + [0 for _ in range(X_val_res.shape[0])]
```

HPC internal usage:
What are the Cluster SLURM Settings and Job Limits?: https://jacksonlaboratory.sharepoint.com/sites/ResearchIT/SitePages/What%20are%20the%20Cluster%20SLURM%20Settings%20and%20Job%20Limits.aspx