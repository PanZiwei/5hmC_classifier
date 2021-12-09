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
input_path='/fastscratch/c-panz/2021-09-13/test'
#df_T4=pd.read_csv(os.path.join(input_path, 'T4.test.bed'), sep='\t')
#df_lambda=pd.read_csv(os.path.join(input_path, 'lambda.test.bed'), sep='\t')
#df_5mClambda=pd.read_csv(os.path.join(input_path, 'lambda_5mC.test.bed'), sep='\t')

# Add the label column
#df_T4['label'] = '5hmC'
#df_lambda['label'] = '5C'
#df_5mClambda['label'] = '5mC'
# Merge the dataset 
#df = pd.concat([df_T4, df_5mClambda, df_lambda])

# Create label column
# Createa a label with representations: 5C = 0, 5mC = 1, 5hmC = 2
#df['label'] = df['label'].map({'5C':0, '5mC':1, '5hmC':2 })
df=pd.read_csv(os.path.join(input_path, 'total.test.bed'), sep='\t')

#Splitting the data into independent and dependent variables
df_feature = df.loc[:,['5hmC_prob','5mC_prob','5C_prob']].values
df_class = df.loc[:,['label']].values
df_class = np.squeeze(df_class) #Convert the label into 1d-array
#print('The probability features set: ')
#print(df_feature.head(5))
#print('The label variable: ')
#print(df_class.head(5))

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
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
#https://medium.com/sfu-cspmp/surviving-in-a-random-forest-with-imbalanced-datasets-b98b963d52eb
#scoring = ('accuracy', 'recall_macro', 'f1_macro')

#Evaluate SRF model
#scores = cross_validate(rf_model, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1)

#Get average evaluation metrics
#print('Mean accuracy: %.3f' % mean(scores['test_accuracy']))
#print('Mean recall_macro: %.3f' % mean(scores['test_recall_macro']))
#print('Mean f1_macro: %.3f' % mean(scores['test_f1_macro']))


score = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
print('Mean f1_macro: %.3f' % mean(score))
print("random forest model ends!")
    