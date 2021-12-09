#! /usr/bin/env python3
"""
Nov 8, 2021, Ziwei Pan
The script is to downsampling all samples with RandomUnderSampler -> random forest with RandomSearch for best parameter
1. Downsampling all samples with idea ratio 0:1:2 = 14:14:1
2. Split the sample into training and testing datasets
3. RandomSearch to find the best parameter on the training dataset
4. Validate the classifer on the remaining fold
5. Save the classification report and confusion matrix plotting for default parameter
"""
import pandas as pd
import numpy as np
import sys, os
import argparse
from datetime import datetime
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


######### Define confusion matrix
#https://gist.github.com/hitvoice/36cf44689065ca9b927431546381a3f7
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
            annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
    
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cmn = pd.DataFrame(cmn, index=label, columns=label)
    cmn.index.name = 'True label'
    cmn.columns.name = 'Predicted label'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cmn, annot=annot, fmt='', ax=ax, cmap="Blues")
    plt.savefig(filename)


#https://stackoverflow.com/questions/50204556/tkinter-tclerror-couldnt-connect-to-display-localhost10-0-when-using-wordc
matplotlib.use('Agg')

if __name__ == '__main__':
    ## Create the parser #
    parser = argparse.ArgumentParser(description="RandomSearch for Megalodon result")
    parser.add_argument('--input_path', required=True, help='input path of fast5 files')
    parser.add_argument('--output_path', required=True, help='output path of fast5 files')
    parser.add_argument('--pkl_model', required=False,
                        default='RandomSearch.noSMOTE.model.pkl',
                        help='model name for best parameter. default RandomSearch.noSMOTE.model.pkl'),
    
    # Execute parse_args()
    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = args.output_path
    pkl_model = args.pkl_model
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
    
    ########## Undersampling the whole dataset
    #Set up undersampling strategy: 0:1:2 = 14:14:1
    n = np.bincount(y)[-1]
    key=[0,1,2]
    value = [(14*n, 14*n, 1*n)]
    for i in value:
        sampling_strategy = dict(zip(key, i)).copy()
        
    sampler = RandomUnderSampler(random_state = 42, sampling_strategy=sampling_strategy)
    X_res, y_res = sampler.fit_resample(X, y)
    print("After downsampling:".format(Counter(y_res)))
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S, %D")
    print("Current Time =", current_time)
    

    ########## Data process
    #Split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_res,
                                                        y_res,
                                                        test_size=0.2,
                                                        stratify=y_res,
                                                        random_state=42)
    print("y_train:{},\n y_test: {}".format(Counter(y_train), Counter(y_test)))
    print("Spliting is done!")

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S, %D")
    print("Current Time =", current_time)

    # Pipeline strategy: rf
    ###Build the pipeline
    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
#    over = SMOTE(sampling_strategy='not majority', random_state = 42, n_jobs = -1)
#    steps = [('o', over), ('m', rf_model)]
    steps = [('m', rf_model)]
    pipeline = imbpipeline(steps=steps)   
    
    # Define cross-validation fold
#    stratified_kfold = StratifiedKFold(n_splits = 3, random_state = 42, shuffle = True)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
          
    #https://stackoverflow.com/questions/51480776/how-to-implement-ratio-based-smote-oversampling-while-cv-ing-dataset
    ##parameter testing
    #assign the parameters to the named step in the pipeline
    #Best parameter from 2021-10-08
    #(k_neighbors=5), n_estimators=80, (max_features = 'auto'), max_depth=5, (min_samples_split=2), min_samples_leaf=2, (bootstrap=True), random_state=42
    #3x6x5x3x2x2=1080
    params = {
#        'o__k_neighbors':[5, 10, 20],      
        'm__n_estimators': [i for i in range(50, 110, 10)], # number of trees in the random forest
        'm__max_features': ['auto'], # number of features in consideration at every split
        'm__max_depth': [i for i in range(5,30,5)], # maximum number of levels allowed in each decision tree
        'm__min_samples_split': [2,5,10], # minimum sample number to split a node
        'm__min_samples_leaf': [1,2], # minimum sample number that can be stored in a leaf node
        'm__bootstrap': [True, False]} # method used to sample data points"
    
    # RandomSearch
    pipe_search = RandomizedSearchCV(estimator = pipeline, 
                                     param_distributions = params, 
                                     scoring='f1_macro', 
                                     cv=cv, 
                                     verbose=3,   
                                     n_iter=100,
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
    joblib.dump(pipe_search.best_estimator_, os.path.join(output_path, pkl_model))
    
    label = ['5C', '5mC', '5hmC']
    ##Save the confusion matrix and plotting
    _confusion_matrix = confusion_matrix(y_true, y_pred)
    _confusion_matrix = pd.DataFrame(_confusion_matrix, index=label, columns=label)
    _confusion_matrix.to_csv(os.path.join(output_path, "RandomSearch.noSMOTE.confusion_matrix.test.csv"), header = True, index= True, sep = ',', float_format='%.4f')
    
    plot_cm(y_test, y_pred, os.path.join(output_path, "RandomSearch.noSMOTE.confusion_matrix.test.png"), label = label, figsize=(10,10))
    
    ##Save the cv result 
    cv_table = pd.DataFrame(pipe_result.cv_results_)
    cv_table.to_csv(os.path.join(output_path, 'RandomSearch.noSMOTE.cv_result.csv'), sep='\t', index = True, header = True, float_format='%.4f')
    
    ## Save the classification report
    report = pd.DataFrame(classification_report(
        y_true, y_pred, target_names=label, output_dict=True)
                         ).transpose()
    report.to_csv(os.path.join(output_path, "RandomSearch.noSMOTE.class_report.test.csv"), header = True, index= True, sep = ',', float_format='%.4f')
    
    print("Saving is done!")
