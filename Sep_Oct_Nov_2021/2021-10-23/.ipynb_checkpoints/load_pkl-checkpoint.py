#! /usr/bin/env python3
"""
Oct 23, 2021, Ziwei Pan
The script is used to: load the model with best parameter set -> load the data -> 
get the cv result on training dataset -> get the  test score/class report/confusion matrix on testing dataset
It is an optimized version of 2021-10-08/read_pkl.py

Usage: python load_pkl.py --input_path $input_path --pkl_path $pkl_path --output_path $output_path
"""

import pandas as pd
import numpy as np
import os, sys
import argparse
from datetime import datetime
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

##Import confusion_matrix.py module
from src.plot_confusion_matrix import plot_cm

import joblib

#https://stackoverflow.com/questions/50204556/tkinter-tclerror-couldnt-connect-to-display-localhost10-0-when-using-wordc
matplotlib.use('Agg')

if __name__ == '__main__':
    ## Create the parser #
    parser = argparse.ArgumentParser(description="Pipeline1 for Megalodon result")
    parser.add_argument('--input_path', required=True, help='input path of feature files')
    parser.add_argument('--pkl_path', required=True, help='input path of model file(.pkl)')
    parser.add_argument('--output_path', required=True, help='output path')
    parser.add_argument('--confusion_matrix', required=False,
                        default='pkl_confusion_matrix_test.png',
                        help='confustion matrix on testing dataset for best parameter. default pkl_confusion_matrix_test.png')
    parser.add_argument('--class_report', required=False,
                        default='pkl_class_report_test.csv',
                        help='class report on testing dataset for best parameter. default pkl_class_report_test.csv')
    
    # Execute parse_args()
    args = parser.parse_args()
    
    input_path = args.input_path
    pkl_path = args.pkl_path
    output_path = args.output_path
    confusion_matrix = args.confusion_matrix
    class_report = args.class_report
    print("Argument is done!")
    
    if pkl_path.endswith(".pkl"):
        model_best = joblib.load(pkl_path)
        print("Model is loading!")
    else:
        print("Wrong model format! Must be .pkl file!")
        
    print("Best model parameter:")
    print(model_best)
    
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
    
    ###########Regenerate the result from save pipeline
    ####Training dataset    
    cv_score = cross_val_score(model_best, X_train, y_train, 
                               cv=stratified_kfold, scoring='f1_macro', n_jobs=-1)
    print("cross-validated metrics on training dataset:{}".format(cv_score))
    print("Mean cross-validated metrics: {}".format(cv_score.mean()))
    
    ####Testing dataset
    y_true, y_pred = y_test, model_best.predict(X_test)
    test_score = model_best.score(X_test, y_test)
    print('Test score:', test_score)
    
    ## Save the classification report
    target_class = ['5C', '5mC', '5hmC']
    report = classification_report(y_test, y_pred, target_names=target_class, output_dict=True)
    report = pd.DataFrame(report).transpose()
    report.to_csv(os.path.join(output_path, class_report), header = True, index= True, sep = ',', float_format='%.4f')
    
    ##Save the confusion matrix   
    plot_cm(y_test, y_pred, os.path.join(output_path, confusion_matrix), labels=None)
    
    print("Saving is done!")