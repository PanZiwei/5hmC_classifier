#! /usr/bin/env python3
"""
Oct 2021, Ziwei Pan
The script is for:
1. Split the sample into training and testing datasets
2. SMOTE to oversample the training dataset
3. Train the the classifier on the training fold for random forest with gridsearch
4. Validate the classifer on the reamining fold
5. Save the classification report and confusion matrix plotting for default parameter
"""
import pandas as pd
import numpy as np
import os, sys
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

#https://stackoverflow.com/questions/50204556/tkinter-tclerror-couldnt-connect-to-display-localhost10-0-when-using-wordc
matplotlib.use('Agg')

#Pre-process the dataset
input_path='/pod/2/li-lab/Ziwei/Nanopore/daily/test/'
#input_path='/pod/2/li-lab/Ziwei/Nanopore/results/megalodon2.3.4_guppy5.0.14'
output_path='/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-13'

#df=pd.read_csv(os.path.join(input_path, 'total.Megalodon.per_read.prob.bed.gz'),compression='gzip', sep='\t')
df=pd.read_csv(os.path.join(input_path, 'total.test.bed.gz'), compression='gzip', sep='\t')
print("Data is loading!")

#Splitting the data into independent and dependent variables
df_feature = df.loc[:,['5hmC_prob','5mC_prob','5C_prob']].values
df_class = df.loc[:,['label']].values
df_class = np.squeeze(df_class) #Convert the label into 1d-array

X = df_feature
y = df_class

#Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=42)
print("Spliting is done!")

now = datetime.now()
current_time = now.strftime("%H:%M:%S, %D")
print("Current Time =", current_time)

# Make the pipeline
pipe_smote_rf = imbpipeline(steps = [('smote', SMOTE(random_state=42)),
                                     ('rfr', RandomForestClassifier(random_state=42))])


# Define parameter for grid search
#Define rf classifier

#Random search result: Best Parameters:  {'classifier__n_estimators': 277, 'classifier__min_samples_split': 10, 'classifier__min_samples_leaf': 2, 'classifier__max_features': 'auto', 'classifier__max_depth': 25, 'classifier__bootstrap': False}  

#5x1x3x2x2x2=120
n_estimators =  [x for x in range(60, 150, 20)] # number of trees in the random forest
max_features = ['auto'] # number of features in consideration at every split
max_depth = [5,10,25] # maximum number of levels allowed in each decision tree
min_samples_split = [2,5] # minimum sample number to split a node
min_samples_leaf = [1,2] # minimum sample number that can be stored in a leaf node
bootstrap = [True, False] # method used to sample data points"


params = {'n_estimators': n_estimators,
          'max_features': max_features,
          'max_depth': max_depth,
          'min_samples_split': min_samples_split,
          'min_samples_leaf': min_samples_leaf,
          'bootstrap': bootstrap}

#assign the parameters to the named step in the pipeline
params = {'rfr__' + key: params[key] for key in params}

# Define cross-validation fold
stratified_kfold = StratifiedKFold(n_splits = 3, random_state = 42, shuffle = True)

##GridSearchCV
rf_random_search = GridSearchCV(estimator = pipe_smote_rf, 
                                param_grid = params, 
                                scoring='f1_macro', 
                                cv=stratified_kfold, 
                                verbose=3, 
                                return_train_score=True, 
                                n_jobs=-1)

rf_result = rf_random_search.fit(X_train, y_train)

y_true, y_pred = y_test, rf_random_search.predict(X_test)


now = datetime.now()
current_time = now.strftime("%H:%M:%S, %D")
print("Current Time =", current_time)
print("Random search is done!")


y_true, y_pred = y_test, rf_random_search.predict(X_test)

cv_score = rf_result.best_score_
test_score = rf_result.score(X_test, y_test)

print ('Best Parameters: ', rf_result.best_params_, ' \n')
print('Best f1_score in cv:', cv_score, '\n')
print('Test score:', test_score)


##Save best parameter
#https://stackoverflow.com/questions/34143829/sklearn-how-to-save-a-model-created-from-a-pipeline-and-gridsearchcv-using-jobli
import joblib
joblib.dump(rf_random_search.best_estimator_, os.path.join(output_path,'rf_smote_gridsearch.pkl'))

df_result = pd.DataFrame(rf_result.cv_results_)
df_result.to_csv(os.path.join(output_path, 'cv_result_smote_gridsearch.tsv'), sep='\t', index = None)

print('Model is saving!')

################# Plot confusion matrix and save the figure
#https://gist.github.com/hitvoice/36cf44689065ca9b927431546381a3f7
def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'True label'
    cm.columns.name = 'Predicted label'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap="Greens")
    plt.savefig(filename)
    
cm_analysis(y_test, y_pred, os.path.join(output_path, 'confusion_matrix_smote_gridsearch.png'),
            labels = None, ymap=None, figsize=(10,10))


## Save the classification report
target_class = ['5C', '5mC', '5hmC']
report = classification_report(y_test, y_pred, target_names=target_class, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv(os.path.join(output_path, 'class_report_smote_gridsearch.csv'), header = True, index= None, sep = ',', float_format='%.4f')
print("Plotting is done!")
