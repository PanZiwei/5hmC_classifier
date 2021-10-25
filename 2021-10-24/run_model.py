#! /usr/bin/env python3
"""
Oct 24, 2021, Ziwei Pan
The script is to repeat the model with the best parameter: set up the model with best parameter set -> load the data -> 
get the cv result on training dataset -> get the  test score/class report/confusion matrix on testing dataset
It is version of 2021-10-23/load_pkl.py

Usage: python load_pkl.py --input_path $input_path --pkl_path $pkl_path --output_path $output_path
"""

import pandas as pd
import numpy as np
import os, sys
import argparse
from datetime import datetime
from collections import Counter
from statistics import mean

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, cross_validate

##Import confusion_matrix.py module
from src.plot_confusion_matrix import plot_cm

#https://stackoverflow.com/questions/50204556/tkinter-tclerror-couldnt-connect-to-display-localhost10-0-when-using-wordc
matplotlib.use('Agg')

if __name__ == '__main__':
    ## Create the parser #
    parser = argparse.ArgumentParser(description="Pipeline1 for Megalodon result")
    parser.add_argument('--input_path', required=True, help='input path of feature files')
    parser.add_argument('--output_path', required=True, help='output path')
    
    # Execute parse_args()
    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = args.output_path
    
    confusion_matrix_test = 'confusion_matrix_test.png'
    scoring_cv = 'scoring_cv.csv'
    class_report_test = 'class_report_test.csv'
    
    print("Argument is done!")
    

    if input_path.endswith(".gz"):
        df=pd.read_csv(input_path, compression='gzip', sep='\t')
        print("Data is loading!")
    else:
        print("Wrong input format! Must be .gz file!")
    
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
    print("Before the pipeline:\n y_train:{},\n y_test: {}".format(Counter(y_train), Counter(y_test)))

    # Define cross-validation fold
    stratified_kfold = StratifiedKFold(n_splits = 3, random_state = 42, shuffle = True)
    
    model_best = imbpipeline(steps=[('o', SMOTE(random_state=42)),
                                    ('m', RandomForestClassifier(max_depth=5, min_samples_leaf=2,n_estimators=80, random_state=42))])
    
    ## Fit the model
    model_best.fit(X_train, y_train)
    
    #https://medium.com/sfu-cspmp/surviving-in-a-random-forest-with-imbalanced-datasets-b98b963d52eb
    scoring = ('accuracy', 'recall_macro', 'f1_macro')
    scores = cross_validate(model_best, X_train, y_train, scoring=scoring, cv=stratified_kfold, n_jobs=-1)
    
    #Get average evaluation metrics
    print('Mean accuracy on cv: %.3f' % mean(scores['test_accuracy']))
    print('Mean recall_macro on cv: %.3f' % mean(scores['test_recall_macro']))
    print('Mean f1_macro on cv: %.3f' % mean(scores['test_f1_macro']))
    
    #Save the scoring matrix on cv to dataframe
    scores = pd.DataFrame.from_dict(scores)
    scores.to_csv(os.path.join(output_path, scoring_cv), header = True, index= None, sep = ',', float_format='%.4f')

    ####Testing dataset
    test_score = model_best.score(X_test, y_test)
    print('Accuracy on testing dataset:', test_score)
    
    y_true, y_pred = y_test, model_best.predict(X_test)

    ## Save the classification report
    target_class = ['5C', '5mC', '5hmC']
    report = classification_report(y_test, y_pred, target_names=target_class, output_dict=True)
    report = pd.DataFrame(report).transpose()
    report.to_csv(os.path.join(output_path, class_report_test), header = True, index= True, sep = ',', float_format='%.4f')
    
    ##Save the confusion matrix   
    plot_cm(y_test, y_pred, os.path.join(output_path, confusion_matrix_test), labels=None)
    print("Saving is done!")