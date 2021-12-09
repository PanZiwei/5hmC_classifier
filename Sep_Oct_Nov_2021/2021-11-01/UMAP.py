#! /usr/bin/env python3
"""
Oct 23, 2021, Ziwei Pan
The script is used to t-SNE and UMAP for Megalodon probability result
Usage: python external_test.py --input_path $input_path --model_path $model_path --output_path $output_path
"""
import pandas as pd
import numpy as np
import os, sys
import argparse

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits import mplot3d
import seaborn as sns

from ipywidgets import IntProgress
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap

#https://stackoverflow.com/questions/50204556/tkinter-tclerror-couldnt-connect-to-display-localhost10-0-when-using-wordc
matplotlib.use('Agg')

def plot_UMAP(input_path, ouput_path):
    df=pd.read_csv(input_path, compression='gzip', sep='\t')
    df_feature = df.loc[:,['5hmC_prob','5mC_prob','5C_prob']].values
    df_class = np.squeeze(df.loc[:,['label']].values) #Covert Pandas DataFrame to 1D array
    
    ##Normalization
    scaled_feature = StandardScaler().fit_transform(df_feature)
    reducer = umap.UMAP(random_state=42,verbose=2)
    embedding = reducer.fit_transform(scaled_feature)
    #https://www.kaggle.com/parulpandey/part3-visualising-kannada-mnist-with-umap

    # Decide color pattern
    cmap = ListedColormap(sns.color_palette("husl",3).as_hex())

    ##Decide category number
    n = len(np.unique(df_class))

    plt.figure(figsize=(8,6))

    plt.scatter(reducer.embedding_[:, 0], 
                reducer.embedding_[:, 1], 
                s= 1, 
                c=df_class, 
                cmap=cmap)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(n+1)-0.5).set_ticks(np.arange(n+1))

    plt.title('Visualizing Megalodon Probablity with UMAP', fontsize=16)
    
    # save
    plt.savefig(os.path.join(output_path,'UMAP_1101.png'))
    
    
if __name__ == '__main__':
    ## Create the parser #
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True, help='input path of feature files')
    parser.add_argument('--output_path', required=True, help='output path')
    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = args.output_path
    
    ###Draw UMAP
    plot_UMAP(input_path, output_path)
    

