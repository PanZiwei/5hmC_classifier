#! /usr/bin/env python3
import pandas as pd
import numpy as np
import sys, os

import joblib
input_path='/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-08'
pkl = 'rf.pkl'
rf_model = joblib.load(os.path.join(input_path, pkl))
print(rf_model)

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


input_path='/pod/2/li-lab/Ziwei/Nanopore/results/megalodon2.3.4_guppy5.0.14'
df=pd.read_csv(os.path.join(input_path, 'total.Megalodon.per_read.prob.bed.gz'),compression='gzip', sep='\t')
print("Data is loading!")

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

rf_result = rf_model.fit(X_train, y_train)

y_true, y_pred = y_test, rf_model.predict(X_test)
test_score = rf_result.score(X_test, y_test)
print('Test score with best parameter:', test_score)