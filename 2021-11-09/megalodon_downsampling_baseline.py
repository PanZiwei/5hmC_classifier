#! /usr/bin/env python3
"""
Nov 9, 2021, Ziwei Pan
The script is to downsampling all the sample -> evaluate Megalodon baseline performance
Based on Megalondon dexcription, 1) P(5hmC) > 0 -> site is 5hmC 2) max(P) -> site is judged
Potential problem: Not sure whether the samples after downsampling is the same or not
"""
import pandas as pd
import numpy as np
import sys, os
import argparse
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

def plot_cm(y_true, y_pred, filename, label = ['5C', '5mC', '5hmC'], figsize=(10,10)):
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
            annot[i, j] = '%.2f%%\n%d/%d' % (p, c, s)
    
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cmn = pd.DataFrame(cmn, index=label, columns=label)
    cmn.index.name = 'True label'
    cmn.columns.name = 'Predicted label'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cmn, annot=annot, fmt='', ax=ax, cmap="Blues", vmin=0, vmax=1) #### The bar is 0~1
    plt.savefig(filename)

    
if __name__ == '__main__':
    ## Create the parser #
    parser = argparse.ArgumentParser(description="Megalodon baseline")
    parser.add_argument('--input_path', required=True, help='input path of fast5 files')
    parser.add_argument('--output_path', required=True, help='output path of fast5 files')
    
    # Execute parse_args()
    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = args.output_path
    print("Argument is done!")
    
    #### Read input
    if input_path.endswith(".gz"):
        df=pd.read_csv(input_path, compression='gzip', sep='\t')
        print("Data is loading!")
    else:
        print("Wrong input format! Must be .gz file!")
        
        
    ######## Downsampling
    #Splitting the data into independent and dependent variables
    X = df.loc[:,['5hmC_prob','5mC_prob','5C_prob']]
    y = df.loc[:,['label']].values
    y = np.squeeze(y) #Convert the label into 1d-array

    n = np.bincount(y)[-1]
    key=[0,1,2]
    value = [(14*n, 14*n, n)] #####Decide the category ratio
    for i in value:
        sampling_strategy = dict(zip(key, i)).copy()

    sampler = RandomUnderSampler(random_state = 42, sampling_strategy=sampling_strategy)
    X_res, y_res = sampler.fit_resample(X, y)
    
    ############ Whole dataset after downsampling
    df = X_res
    y_true = y_res
    
    ### three way judge: https://stackoverflow.com/a/20033232
    max_conditions = [
        (df[['5hmC_prob','5mC_prob','5C_prob']].max(axis=1) == df['5hmC_prob']),
        (df[['5hmC_prob','5mC_prob','5C_prob']].max(axis=1) == df['5mC_prob']),
        (df[['5hmC_prob','5mC_prob','5C_prob']].max(axis=1) == df['5C_prob'])
        ]
    # create a list of the values we want to assign for each condition
    values = [2, 1, 0]
    # create a new column and use np.select to assign values to it using our lists as arguments
    y_pred = np.select(max_conditions, values)
    
    ##Save the confusion matrix and plotting
    label = ['5C', '5mC', '5hmC']
    _confusion_matrix = confusion_matrix(y_true, y_pred)
    _confusion_matrix = pd.DataFrame(_confusion_matrix, index=label, columns=label)
    _confusion_matrix.to_csv(os.path.join(output_path, "Megalodon.downsampling.total.max_value.csv"), header = True, index= True, sep = ',', float_format='%.4f')
    
    plot_cm(y_true, y_pred, os.path.join(output_path, "Megalodon.downsampling.total.max_value.png"), label = label, figsize=(10,10))

    print('f1_macro for three-way maximum:',f1_score(y_true, y_pred, average='macro'))
    
    ## Save the classification report
    report = pd.DataFrame(classification_report(
        y_true, y_pred, target_names=label, output_dict=True)
                         ).transpose()
    report.to_csv(os.path.join(output_path, "Megalodon.downsampling.total.max_value.csv"), header = True, index= None, sep = ',', float_format='%.4f')
    del y_pred

          
    ### P(5hmC) > 0 cut-off
    ### # create a list of condtion
    conditions = [
        (df['5hmC_prob'] > 0),
        (df['5hmC_prob'] <= 0) & (df['5mC_prob'] > df['5C_prob']),
        (df['5hmC_prob'] <= 0) & (df['5mC_prob'] <= df['5C_prob'])
        ]
    # create a list of the values we want to assign for each condition
    values = [2, 1, 0]
    # create a new column and use np.select to assign values to it using our lists as arguments
    y_pred = np.select(conditions, values)
    
    ##Save the confusion matrix and plotting
    label = ['5C', '5mC', '5hmC']
    _confusion_matrix = confusion_matrix(y_true, y_pred)
    _confusion_matrix = pd.DataFrame(_confusion_matrix, index=label, columns=label)
    _confusion_matrix.to_csv(os.path.join(output_path, "Megalodon.downsampling.total.zero_cutoff.csv"), header = True, index= True, sep = ',', float_format='%.4f')
    
    plot_cm(y_true, y_pred, os.path.join(output_path, "Megalodon.downsampling.total.zero_cutoff.png"), label = label, figsize=(10,10))
    
    print('f1_macro for zero cutoff:', f1_score(y_true, y_pred, average='macro'))
    
    
    ## Save the classification report
    report = pd.DataFrame(classification_report(
        y_true, y_pred, target_names=label, output_dict=True)
                         ).transpose()
    report.to_csv(os.path.join(output_path, "Megalodon.downsampling.total.zero_cutoff.csv"), header = True, index= None, sep = ',', float_format='%.4f')
    del y_pred
    
    del df, y_true
    
    print("Downsampling is done!")
    
    ############# Testing dataset after downsampling
    #Split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_res,
                                                        y_res,
                                                        test_size=0.2,
                                                        stratify=y_res,
                                                        random_state=42)
    
    df = X_test
    y_true = y_test
    
    ### three way judge: https://stackoverflow.com/a/20033232
    max_conditions = [
        (df[['5hmC_prob','5mC_prob','5C_prob']].max(axis=1) == df['5hmC_prob']),
        (df[['5hmC_prob','5mC_prob','5C_prob']].max(axis=1) == df['5mC_prob']),
        (df[['5hmC_prob','5mC_prob','5C_prob']].max(axis=1) == df['5C_prob'])
        ]
    # create a list of the values we want to assign for each condition
    values = [2, 1, 0]
    # create a new column and use np.select to assign values to it using our lists as arguments
    y_pred = np.select(max_conditions, values)
    
    ##Save the confusion matrix and plotting
    label = ['5C', '5mC', '5hmC']
    _confusion_matrix = confusion_matrix(y_true, y_pred)
    _confusion_matrix = pd.DataFrame(_confusion_matrix, index=label, columns=label)
    _confusion_matrix.to_csv(os.path.join(output_path, "Megalodon.downsampling.test.max_value.csv"), header = True, index= True, sep = ',', float_format='%.4f')
    
    plot_cm(y_true, y_pred, os.path.join(output_path, "Megalodon.downsampling.test.max_value.png"), label = label, figsize=(10,10))

    print('f1_macro for three-way maximum:',f1_score(y_true, y_pred, average='macro'))
    
    ## Save the classification report
    report = pd.DataFrame(classification_report(
        y_true, y_pred, target_names=label, output_dict=True)
                         ).transpose()
    report.to_csv(os.path.join(output_path, "Megalodon.downsampling.test.max_value.csv"), header = True, index= None, sep = ',', float_format='%.4f')
    del y_pred

          
    ### P(5hmC) > 0 cut-off
    ### # create a list of condtion
    conditions = [
        (df['5hmC_prob'] > 0),
        (df['5hmC_prob'] <= 0) & (df['5mC_prob'] > df['5C_prob']),
        (df['5hmC_prob'] <= 0) & (df['5mC_prob'] <= df['5C_prob'])
        ]
    # create a list of the values we want to assign for each condition
    values = [2, 1, 0]
    # create a new column and use np.select to assign values to it using our lists as arguments
    y_pred = np.select(conditions, values)
    
    ##Save the confusion matrix and plotting
    label = ['5C', '5mC', '5hmC']
    _confusion_matrix = confusion_matrix(y_true, y_pred)
    _confusion_matrix = pd.DataFrame(_confusion_matrix, index=label, columns=label)
    _confusion_matrix.to_csv(os.path.join(output_path, "Megalodon.downsampling.test.zero_cutoff.csv"), header = True, index= True, sep = ',', float_format='%.4f')
    
    plot_cm(y_true, y_pred, os.path.join(output_path, "Megalodon.downsampling.test.zero_cutoff.png"), label = label, figsize=(10,10))
    
    print('f1_macro for zero cutoff:', f1_score(y_true, y_pred, average='macro'))
    
    
    ## Save the classification report
    report = pd.DataFrame(classification_report(
        y_true, y_pred, target_names=label, output_dict=True)
                         ).transpose()
    report.to_csv(os.path.join(output_path, "Megalodon.downsampling.test.zero_cutoff.csv"), header = True, index= None, sep = ',', float_format='%.4f')
    del y_pred
    
    
    
    
