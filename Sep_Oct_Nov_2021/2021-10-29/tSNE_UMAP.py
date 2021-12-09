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


def plot_tsne(input_path, output_path):
    df=pd.read_csv(input_path, compression='gzip', sep='\t')
    df_feature = df.loc[:,['5hmC_prob','5mC_prob','5C_prob']].values
    df_class = np.squeeze(df.loc[:,['label']].values) #Covert Pandas DataFrame to 1D array
    
    tsne_feature = TSNE(n_components=2, verbose=2, perplexity=50, random_state=42).fit_transform(df_feature) 
    
    #https://medium.com/@violante.andre/an-introduction-to-t-sne-with-python-example-47e6ae7dc58f
    ##Store tsne result into a dataframe
    tsne_df = pd.DataFrame({
        'X': tsne_feature[:,0], #5hmC
        'Y': tsne_feature[:,1], #5mC
        'label': df_class
    })
    
    #Set up figure size
    plt.figure(figsize=(8,6))

    plt.scatter(x=tsne_df["X"],
                y=tsne_df["Y"],
                c=tsne_df["label"],
                s=1,
                cmap='rainbow')
    plt.gca().set_aspect('equal', 'datalim')

    ##Decide category number
    n = len(np.unique(tsne_df["label"]))

    plt.colorbar(boundaries=np.arange(n+1)-0.5).set_ticks(np.arange(n+1))

    plt.xlabel('5hmC', weight = 'bold')
    plt.ylabel('5mC', weight = 'bold')
    plt.title('t-SNE for Megalodon probability result', fontsize=16)

    # save
    plt.savefig(os.path.join(output_path,'tsne_2D.png'))
    
    
#3D tSNE    
def plot_tsne_3D(input_path, output_path):
    df=pd.read_csv(input_path, compression='gzip', sep='\t')
    df_feature = df.loc[:,['5hmC_prob','5mC_prob','5C_prob']].values
    df_class = np.squeeze(df.loc[:,['label']].values) #Covert Pandas DataFrame to 1D array
    
    #Normalization
    tsne_feature_3D = TSNE(n_components=3, verbose=1, perplexity=50, random_state=42).fit_transform(df_feature) 
    
    #https://medium.com/@violante.andre/an-introduction-to-t-sne-with-python-example-47e6ae7dc58f
    ##Store tsne result into a dataframe
    tsne_df_3D = pd.DataFrame({
        'X': tsne_feature_3D[:,0], #5hmC
        'Y': tsne_feature_3D[:,1], #5mC
        'Z': tsne_feature_3D[:,2],
        'label': df_class
    })
    
   ##Create figure and axes
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111, projection = '3d')

    # get colormap from seaborn
    cmap = ListedColormap(sns.color_palette("husl",3).as_hex())

    sc = ax.scatter(
        xs=tsne_df_3D['X'], 
        ys=tsne_df_3D['Y'], 
        zs=tsne_df_3D['Z'],
        c=tsne_df_3D['label'],  #label
        cmap=cmap #color pattern from seaborn
    )

    ax.set_xlabel('5hmC')
    ax.set_ylabel('5mC')
    ax.set_zlabel('5C')
    ax.set_title('3D t-SNE for Megalodon probablity result').set_fontsize('16')

    # legend
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
#    plt.show()

    # save
    plt.savefig(os.path.join(output_path,'tsne_3D.png'))
    

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
    plt.savefig(os.path.join(output_path,'UMAP.png'))
    
    
if __name__ == '__main__':
    ## Create the parser #
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True, help='input path of feature files')
    parser.add_argument('--output_path', required=True, help='output path')
    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = args.output_path
    
    ###Draw tsne
    plot_tsne(input_path, output_path)

    ###Draw tsne_3D
    plot_tsne_3D(input_path, output_path)
    
    ###Draw UMAP
    plot_UMAP(input_path, output_path)
    

