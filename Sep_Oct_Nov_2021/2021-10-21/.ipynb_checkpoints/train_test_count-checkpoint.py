#! /usr/bin/env python3
"""
Oct 21, 2021, Ziwei Pan
The script is used count the file number train/test datasets after train_test_split
Usage: python3 train_test_count.py > train_test.txt
"""
input_path = '/pod/2/li-lab/Ziwei/Nanopore/results/megalodon2.3.4_guppy5.0.14/total.Megalodon.per_read.prob.bed.gz'

import pandas as pd
import numpy as np
import os, sys
import argparse
from datetime import datetime
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold

now = datetime.now()
current_time = now.strftime("%H:%M:%S, %D")
print("Current Time =", current_time)

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
