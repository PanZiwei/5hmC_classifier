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
import argparse
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

from plot_confusion_matrix import cm_analysis

#https://stackoverflow.com/questions/50204556/tkinter-tclerror-couldnt-connect-to-display-localhost10-0-when-using-wordc
matplotlib.use('Agg')

if __name__ == '__main__':
    ## Create the parser #
    parser = argparse.ArgumentParser(description="random forest pipeline for Megalodonr result")
    parser.add_argument('--input_path', required=True, help='input path of fast5 files')
    parser.add_argument('--output_path', required=True, help='output path of fast5 files')
    
    parser.add_argument('--pkl_model', required=False,
                        default='model.pkl',
                        help='pkl model name for best parameter. default model.pkl'),
    parser.add_argument('--confusion_matrix', required=False,
                        default='Confusion_matrix.png',
                        help='confustion matrix name for best parameter. default Confusion_matrix.png')
    parser.add_argument('--cv_result', required=False,
                        default='cv_result.csv',
                        help='cv table name for best parameter. default cv_result.csv'),
    parser.add_argument('--class_report', required=False,
                        default='class_report.csv',
                        help='class report for best parameter. default class_report.csv')
    
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

    ################Save the result
    
    ##Save best parameter
    #https://stackoverflow.com/questions/34143829/sklearn-how-to-save-a-model-created-from-a-pipeline-and-gridsearchcv-using-jobli
    import joblib
    joblib.dump(rf_random_search.best_estimator_, os.path.join(output_path,pkl_model))
    
    
    ##Save the confusion matrix   
    cm_analysis(y_test, y_pred, os.path.join(output_path, cm_png), labels = None, ymap=None, figsize=(10,10))
    
    
    ##Save the cv result 
    df_result = pd.DataFrame(rf_result.cv_results_)
    df_result.to_csv(os.path.join(output_path, cv_result), sep='\t', index = None)

    
    ## Save the classification report
    target_class = ['5C', '5mC', '5hmC']
    report = classification_report(y_test, y_pred, target_names=target_class, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(os.path.join(output_path, class_report), header = True, index= None, sep = ',', float_format='%.4f')
    
    print("Saving is done!")
