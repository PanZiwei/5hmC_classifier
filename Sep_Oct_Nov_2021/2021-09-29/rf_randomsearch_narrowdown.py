#! /usr/bin/env python3
"""
Sep2021, Ziwei Pan
The script is for:
1. Apply SMOTE to oversampling the imbalanced datasets
2. Build random forest with default parameter
3. Save the classification report and confusion matrix plotting for default parameter
"""
import pandas as pd
import numpy as np
import sys, os, logging
from datetime import datetime
    

#Pre-process the dataset
#input_path='/pod/2/li-lab/Ziwei/Nanopore/daily/test/'
input_path='/pod/2/li-lab/Ziwei/Nanopore/results/megalodon2.3.4_guppy5.0.14'
output_path='/pod/2/li-lab/Ziwei/Nanopore/daily/2021-09-29'

df=pd.read_csv(os.path.join(input_path, 'total.Megalodon.per_read.prob.bed.gz'),compression='gzip', sep='\t')
#df=pd.read_csv(os.path.join(input_path, 'total.test.bed'), sep='\t')
print("Data is loading!")

#Splitting the data into independent and dependent variables
df_feature = df.loc[:,['5hmC_prob','5mC_prob','5C_prob']].values
df_class = df.loc[:,['label']].values
df_class = np.squeeze(df_class) #Convert the label into 1d-array


from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedStratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter

#Randomly spilt dataset to traing/testing dataset with the original ratio
X_train, X_test, y_train, y_test = train_test_split(df_feature, 
                                                    df_class, 
                                                    test_size=0.2, 
                                                    stratify=df_class)
print("Spliting is done!")

now = datetime.now()
current_time = now.strftime("%H:%M:%S, %D")
print("Current Time =", current_time)

######## SMOTE oversampling
sm = SMOTE(random_state=42)
counter = Counter(y_train)
print("Before SMOTE: {}".format(counter))

X_train_new, y_train_new = sm.fit_resample(X_train, y_train)
counter = Counter(y_train_new)
print("After SMOTE: {}".format(counter))


now = datetime.now()
current_time = now.strftime("%H:%M:%S, %D")
print("Current Time =", current_time)


# Define model
rf_model = RandomForestClassifier(random_state=42)

# define evaluation
#cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

#Define parameter
#10x2x5x2x2x2=800
n_estimators =  [int(x) for x in np.linspace(start = 80, stop = 120, num = 10)] # number of trees in the random forest
max_features = ['auto'] # number of features in consideration at every split
max_depth = [int(x) for x in np.linspace(start = 10, stop = 40, num = 5)] # maximum number of levels allowed in each decision tree
min_samples_split = [2, 5, 10] # minimum sample number to split a node
min_samples_leaf = [1, 2] # minimum sample number that can be stored in a leaf node
bootstrap = [True, False] # method used to sample data points

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


#Create Stratified K-fold cross validation
#cv = RepeatedStratifiedKFold(n_splits=5 n_repeats=3, random_state=1)


##random search
rf_random = RandomizedSearchCV(estimator = rf_model, 
                               param_distributions = random_grid, 
                               scoring='f1_macro', 
                               cv=5, 
                               verbose=5, 
                               return_train_score=True, 
                               n_iter = 10,  #The number of parameter settings that are tried 
                               random_state=35,
                               n_jobs=-1)

rf_result = rf_random.fit(X_train_new, y_train_new)

# print the best parameters
print ('Best Parameters: ', rf_random.best_params_, ' \n')

print('Bestf1_score:', rf_random.best_score_, '\n')

now = datetime.now()
current_time = now.strftime("%H:%M:%S, %D")
print("Current Time =", current_time)

y_true, y_pred = y_test, rf_random.predict(X_test)

now = datetime.now()
current_time = now.strftime("%H:%M:%S, %D")
print("Current Time =", current_time)

##Save best parameter
#https://stackoverflow.com/questions/34143829/sklearn-how-to-save-a-model-created-from-a-pipeline-and-gridsearchcv-using-jobli
import joblib
joblib.dump(rf_random.best_estimator_, os.path.join(output_path,'rf_narrowdown.pkl'))

df_result = pd.DataFrame(rf_result.cv_results_)
df_result.to_csv(os.path.join(output_path, 'cv_narrowdown.tsv'), sep='\t', index = None)

now = datetime.now()
current_time = now.strftime("%H:%M:%S, %D")
print("Current Time =", current_time)

print('Model is saving!')

################# Plot confusion matrix and save the figure
#https://gist.github.com/hitvoice/36cf44689065ca9b927431546381a3f7
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
    
cm_analysis(y_test, y_pred, os.path.join(output_path, 'confusion_matrix_narrowdown.png'),
            labels = None, ymap=None, figsize=(10,10))


## Save the classification report
from sklearn.metrics import classification_report
target_class = ['5C', '5mC', '5hmC']
report = classification_report(y_test, y_pred, target_names=target_class, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv(os.path.join(output_path, 'class_report_narrowdown.csv'), header = True, index= None, sep = ',', float_format='%.4f')
print("Plotting is done!")
