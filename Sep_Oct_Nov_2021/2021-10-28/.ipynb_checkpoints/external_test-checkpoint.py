#! /usr/bin/env python3
"""
Oct 23, 2021, Ziwei Pan
The script is used for external testing
load the model -> load the data -> save the test score/class report/confusion matrix/confusion matrix visualization on testing dataset
Usage: python external_test.py --input_path $input_path --model_path $model_path --output_path $output_path
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

import joblib

#https://stackoverflow.com/questions/50204556/tkinter-tclerror-couldnt-connect-to-display-localhost10-0-when-using-wordc
matplotlib.use('Agg')

def plot_cm(y_true, y_pred, filename, label=['5C', '5mC', '5hmC'], figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with annotations.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    cm = confusion_matrix(y_true, y_pred)
    # Normalise
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            s = cm_sum[i]
            annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
    
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cmn = pd.DataFrame(cmn, index=label, columns=label)
    cmn.index.name = 'True label'
    cmn.columns.name = 'Predicted label'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cmn, annot=annot, fmt='', ax=ax, cmap="Blues")
    plt.savefig(filename)

    
if __name__ == '__main__':
    ## Create the parser #
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True, help='input path of feature files')
    parser.add_argument('--output_path', required=True, help='output path')
    parser.add_argument('--model_path', required=True, help='input path of model file(.pkl)')
    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = args.output_path
    model_path = args.model_path
    
    label=['5C', '5mC', '5hmC']
    dataset = 'test'
    
    if model_path.endswith(".pkl"):
        model = joblib.load(model_path)
        print("Model is loading!")
    else:
        print("Wrong model format! Must be .pkl file!")
        
    ### Load the dataset
    df=pd.read_csv(input_path, compression='gzip', sep='\t')
     
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
    print("Before the pipeline:\n y_train:{},\n y_test: {}".format(Counter(y_train), Counter(y_test)))
    
    
    # Define cross-validation fold
    stratified_kfold = StratifiedKFold(n_splits = 3, random_state = 42, shuffle = True)

    
    ###########Regenerate the result from save pipeline
    ####Training dataset    
    cv_score = cross_val_score(model, X_train, y_train, 
                               cv=stratified_kfold, scoring='f1_macro', n_jobs=-1)
    print("cross-validated metrics on training dataset:{}".format(cv_score))
    print("Mean cross-validated metrics: {}".format(cv_score.mean()))
    
    ####Testing dataset
    y_true, y_pred = y_test, model.predict(X_test)
    test_score = model.score(X_test, y_test)
    print('Test score:', test_score)
    
    ## Save the classification report
    class_report = "class_report_{}.csv".format(dataset)
    report = pd.DataFrame(classification_report(
        y_true, y_pred, target_names=label, output_dict=True)
                         ).transpose()
    report.to_csv(os.path.join(output_path, class_report), header = True, index= True, sep = ',', float_format='%.4f')
    
    ##Save the confusion matrix and plotting
    matrix_name = "confusion_matrix_{}.csv".format(dataset)
    _confusion_matrix = confusion_matrix(y_true, y_pred)
    _confusion_matrix = pd.DataFrame(_confusion_matrix, index=label,columns=label)
    _confusion_matrix.to_csv(os.path.join(output_path, matrix_name), header = True, index= True, sep = ',', float_format='%.4f')
    
    matrix_png = "confusion_matrix_{}.png".format(dataset)
    plot_cm(y_true, y_pred, os.path.join(output_path, matrix_png), label=['5C', '5mC', '5hmC'], figsize=(10,10))
    
    print("saving is done!")
    

