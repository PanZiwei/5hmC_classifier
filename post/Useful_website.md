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

Save the best parameter; https://stackoverflow.com/questions/34143829/sklearn-how-to-save-a-model-created-from-a-pipeline-and-gridsearchcv-using-jobli
```python
import joblib
joblib.dump(rf_random_search.best_estimator_, os.path.join(output_path,pkl_model))
```