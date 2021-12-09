#! /usr/bin/env python3
"""
Sep2021, Ziwei Pan
Random forest model for Megalodon label
"""
import pandas as pd
import numpy as np
import sys, os, logging
    
logging.basicConfig(format='%(asctime)s - [%(filename)s:%(lineno)d] - %(levelname)s: %(message)s', level=logging.INFO)
logging.info("The script begins!")

#Pre-process the dataset
input_path='/fastscratch/c-panz/2021-09-13'
output_path='/pod/2/li-lab/Ziwei/Nanopore/daily/2021-09-14'

df=pd.read_csv(os.path.join(input_path, 'total.Megalodon.per_read.prob.bed'), sep='\t')
print("Data is loading!")

#Splitting the data into independent and dependent variables
df_feature = df.loc[:,['5hmC_prob','5mC_prob','5C_prob']].values
df_class = df.loc[:,['label']].values
df_class = np.squeeze(df_class) #Convert the label into 1d-array
#print('The probability features set: ')
#print(df_feature.head(5))
#print('The label variable: ')
#print(df_class.head(5))

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from statistics import mean

print("random forest model begins!")
# https://medium.com/sfu-cspmp/surviving-in-a-random-forest-with-imbalanced-datasets-b98b963d52eb
# Build the random forest model
example_params = {
    'n_estimators': 10,
     'max_depth': 5,
    'random_state': 42
    }

rf_model = RandomForestClassifier(**example_params)

#Create Stratified K-fold cross validation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

#Randomly spilt dataset to traing/testing dataset with the original ratio
X_train, X_test, y_train, y_test = train_test_split(df_feature, 
                                                    df_class, 
                                                    test_size=0.2, 
                                                    stratify=df_class)
print("Spliting is done!")


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

rf_model = RandomForestClassifier(random_state=42)
#Create Stratified K-fold cross validation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# 5 * 2 * 4 * 3 * 3 = 360

param_grid = {
    'n_estimators'     : [10, 50, 100, 200, 300],
    'max_features'     : ['sqrt', None],
    'max_depth'        : [5, 10, 25, 50],
    'min_samples_split': [5, 8, 10], #The minimum sample number to split an internal node
    'min_samples_leaf' : [3, 4, 5],
    'bootstrap'        : [True, False]
}

scores = ['accuracy', 'recall_macro', 'f1_macro']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    ##Grid search
    clf = GridSearchCV(estimator = rf_model, param_grid = param_grid, scoring= score, cv=cv, 
                       verbose=0, return_train_score=True, n_jobs=-1)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
