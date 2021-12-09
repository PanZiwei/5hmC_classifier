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
    
logging.basicConfig(format='%(asctime)s - [%(filename)s:%(lineno)d] - %(levelname)s: %(message)s', level=logging.INFO)
logging.info("The script begins!")

#Pre-process the dataset
input_path='/pod/2/li-lab/Ziwei/Nanopore/daily/2021-09-13'
output_path='/pod/2/li-lab/Ziwei/Nanopore/daily/2021-09-27/file'

df=pd.read_csv(os.path.join(input_path, 'total.Megalodon.per_read.prob.bed.gz'),compression='gzip', sep='\t')
print("Data is loading!")

#Splitting the data into independent and dependent variables
df_feature = df.loc[:,['5hmC_prob','5mC_prob','5C_prob']].values
df_class = df.loc[:,['label']].values
df_class = np.squeeze(df_class) #Convert the label into 1d-array


from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, cross_validate
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

#    'n_estimators'     : [10, 50, 100],
#    'max_depth'        : [10, 25],
#    'min_samples_split': [5, 10],
        
#define model
example_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': 42
    }
rf_model = RandomForestClassifier(**example_params)

#Create Stratified K-fold cross validation
cv = RepeatedStratifiedKFold(n_splits=5, random_state=1)

#Evalute pipeline
f1_macro = cross_val_score(rf_model, X_train_new, y_train_new, cv=cv, scoring='f1_macro', verbose=3, n_jobs=-1)
print('Mean f1_macro:{:.3f}'.format(np.mean(f1_macro)))

#Fitting the model
rf_model.fit(X_train_new, y_train_new)
y_pred = rf_model.predict(X_test)
print("Training is done!")

now = datetime.now()
current_time = now.strftime("%H:%M:%S, %D")
print("Current Time =", current_time)

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
    
cm_analysis(y_test, y_pred, os.path.join(output_path, 'confusion_matrix_100_10_5.png'),
            labels = None, ymap=None, figsize=(10,10))


## Save the classification report
from sklearn.metrics import classification_report
target_class = ['5C', '5mC', '5hmC']
report = classification_report(y_test, y_pred, target_names=target_class, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv(os.path.join(output_path, 'class_report_100_10_5.csv'), header = True, index= None, sep = ',', float_format='%.4f')
print("Plotting is done!")
