#! /usr/bin/env python3
"""
Oct 26, 2021, Ziwei Pan
The script is to find best unsampling strategy with best parameter: Build a pipeline: RandomUnderSampler -> SMOTE -> random forest
1. Split the sample into training and testing datasets
2. Build the pipeline: RandomUnderSampler to downsamping the training dataset, then SMOTE to oversample the training dataset, random forest
3. RandomSearch to find the best parameter on the training dataset
4. Validate the classifer on the remaining fold
5. Save the classification report and confusion matrix plotting for default parameter
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

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

##Import confusion_matrix.py module
from src.plot_confusion_matrix import plot_cm

#https://stackoverflow.com/questions/50204556/tkinter-tclerror-couldnt-connect-to-display-localhost10-0-when-using-wordc
matplotlib.use('Agg')

if __name__ == '__main__':
    ## Create the parser #
    parser = argparse.ArgumentParser(description="Pipeline1 for Megalodon result")
    parser.add_argument('--input_path', required=True, help='input path of fast5 files')
    parser.add_argument('--output_path', required=True, help='output path of fast5 files')
    parser.add_argument('--pkl_model', required=False,
                        default='p1_model.pkl',
                        help='pkl model name for best parameter. default p1_model.pkl'),
    parser.add_argument('--confusion_matrix', required=False,
                        default='p1_confusion_matrix.png',
                        help='confustion matrix name for best parameter. default p1_confusion_matrix.png')
    parser.add_argument('--cv_result', required=False,
                        default='p1_cv_result.csv',
                        help='cv table name for best parameter. default p1_cv_result.csv'),
    parser.add_argument('--class_report', required=False,
                        default='p1_class_report.csv',
                        help='class report for best parameter. default p1_class_report.csv')
    
    # Execute parse_args()
    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = args.output_path
    pkl_model = args.pkl_model
    cm_png = args.confusion_matrix
    cv_result = args.cv_result
    class_report = args.class_report

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
    print("Before the pipeline:\n y_train:{},\n y_test: {}".format(Counter(y_train), Counter(y_test)))
    print("Spliting is done!")

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S, %D")
    print("Current Time =", current_time)

    # Pipeline strategy: RandomUnderSampler -> SMOTE -> rf
    ###Build the pipeline
    rf_model = RandomForestClassifier(max_depth=5, min_samples_leaf=2,n_estimators=80, random_state=42)
    under = RandomUnderSampler(random_state = 42)
    over = SMOTE(sampling_strategy='not majority', random_state = 42, n_jobs = -1)
    steps = [('u', under), ('o', over), ('m', rf_model)]

    pipeline = imbpipeline(steps=steps)

    #####Prepare downsamping_strategy
    value_sub1 = [(9*i*1108, 14*i*1108, 1108) for i in range(2,40,2)]
    value_sub2 = [(36*i*1108, 56*i*1108, 1108) for i in range(10,110,5)]
    value = list()
    value = value_sub1 + value_sub2
    key=[0,1,2]
    sampling_strategy = list()
    for i in value:
        item = dict(zip(key, i)).copy()
        sampling_strategy.append(item)

    
    #https://stackoverflow.com/questions/51480776/how-to-implement-ratio-based-smote-oversampling-while-cv-ing-dataset
    ##parameter testing
    #assign the parameters to the named step in the pipeline
    params = {
        'u__sampling_strategy' : sampling_strategy} 
          

    # Define cross-validation fold
    stratified_kfold = StratifiedKFold(n_splits = 3, random_state = 42, shuffle = True)
    
    # RandomSearch
    pipe_search = GridSearchCV(estimator = pipeline, 
                               param_grid = params, 
                               scoring='f1_macro', 
                               cv=stratified_kfold, 
                               verbose=3, 
                               return_train_score=True, 
                               n_jobs=-1)

    pipe_result = pipe_search.fit(X_train, y_train)
    
    y_true, y_pred = y_test, pipe_search.predict(X_test)
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S, %D")
    print("Current Time =", current_time)
    print("Random search is done!")

    cv_score = pipe_result.best_score_
    test_score = pipe_result.score(X_test, y_test)

    print ('Best Parameters: ', pipe_result.best_params_, ' \n')
    print('Best f1_score in cv:', cv_score, '\n')
    print('Test score:', test_score)

    ################Save the result
    
    ##Save best parameter
    #https://stackoverflow.com/questions/34143829/sklearn-how-to-save-a-model-created-from-a-pipeline-and-gridsearchcv-using-jobli
    import joblib
    joblib.dump(pipe_search.best_estimator_, os.path.join(output_path,pkl_model))
    
    ##Save the cv result 
    cv_table = pd.DataFrame(pipe_result.cv_results_)
    cv_table.to_csv(os.path.join(output_path, cv_result), sep='\t', index = True)
    
    ##Save the confusion matrix   
    plot_cm(y_test, y_pred, os.path.join(output_path, cm_png), labels=None)
    
    ## Save the classification report
    target_class = ['5C', '5mC', '5hmC']
    report = classification_report(y_test, y_pred, target_names=target_class, output_dict=True)
    report = pd.DataFrame(report).transpose()
    report.to_csv(os.path.join(output_path, class_report), header = True, index= None, sep = ',', float_format='%.4f')
    
    print("Saving is done!")
