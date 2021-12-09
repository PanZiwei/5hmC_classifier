#! /usr/bin/env python3
input_path='/fastscratch/c-panz/2021-09-02'
output_path='/pod/2/li-lab/Ziwei/Nanopore/daily/2021-09-02'

cutoff=[x/10 for x in range(0, 11)]

#Draw the 5hmC_freq distribution
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
def methyl_freq_plotting(input_path, input_file, prefix, cutoff, output_path):
    #Plot points(marker pattern, size)
    df = pd.read_csv(os.path.join(input_path, input_file), sep='\t')
    x=df['5hmC_freq']*100
    binwidth=5
    kwargs = dict(alpha=1, bins=np.arange(0, 100+binwidth, binwidth),  ##Set up the x_axis
                  edgecolor='black', align='mid')
    plt.figure(figsize=(6,4), dpi=90)
    plt.gca().set(title='{} Histogram of 5hmC% by Megalodon(cutoff={})'.format(prefix, cutoff),
                  xlabel='%5hmC per base', ylabel='Frequency')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    
    if len(x) == 0:
        return None
        
    elif len(x) !=0:
        n, bins, _ = plt.hist(x=x, **kwargs)
    
        k = [round(i/len(x)*100, 1) for i in n]
        for i in range(0, len(n)):
            x_pos = bins[i] - 0.5
            y_pos = n[i] 
            label = str(k[i]) # relative frequency of each bin
            plt.gca().text(x_pos, y_pos, label, fontsize='small')
        
    fig_name = '{}.Megalodon.per_site.{}'.format(prefix,cutoff) + '.png'   
    plt.savefig(fname=os.path.join(output_path, fig_name), 
                    dpi=100, bbox_inches='tight', pad_inches=0.0, facecolor='w', transparent=False)

    plt.show()

#APL
#Pre_read result: 5mC_lambda.Megalodon.per_site.0.1.tsv
prefix='APL'
for i in cutoff:
    print(i)
    methyl_freq_plotting(input_path, input_file='{}.Megalodon.per_site.{}.tsv'.format(prefix, float(i)), prefix=prefix, cutoff=i, output_path=output_path)
    