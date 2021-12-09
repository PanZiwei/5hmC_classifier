#! /usr/bin/env python3
## The script is used to extract CG-centered feature from original feature files
## Right now kmer=17
import pandas as pd
import numpy as np
import os

input_path='/pod/2/li-lab/Ziwei/Nanopore/results/feature_Guppy5.0.11'

kmer = 17
num_bases = (kmer - 1) // 2

names = ['chrom', 'site_pos', 'align_strand', 'loc_in_ref', 'read_id', 'read_strand',
         'kmer_seq', 'kmer_signal_mean', 'kmer_signal_std', 'kmer_signal_length', 'kmer_signal_range',
         'label']
####site_pos: start position in 0-based,
##if alignd_strand = +, the pos is the C or G location in the reference, pos = [site_pos, site_pos + 1]
##if alignd_strand = -, the pos is the G or C location in the reference (C in the kmer still), pos = [site_pos, site_pos + 1]
##Filter is in need to pick up the CG pattern

###5hmC
df=pd.read_csv(os.path.join(input_path,'T4_5hmC.csv'),sep='\t', header = None)
df.columns = names
####Extract 5hmC
CG_index = list()
for seq in df['kmer_seq']:
    if seq[num_bases:num_bases+2] == 'CG':
        CG_index.append(seq) 
df_selected = df[df['kmer_seq'].isin(CG_index)]
df_selected.to_csv(os.path.join(input_path, 'T4_5hmC.CG.csv'), header = None, index = None, sep='\t')
del df_selected, CG_index, df

###5mC
df=pd.read_csv(os.path.join(input_path,'lambda_5mC.csv'),sep='\t', header = None)
df.columns = names
                   
####Extract 5mC
CG_index = list()
for seq in df['kmer_seq']:
    if seq[num_bases:num_bases+2] == 'CG':
        CG_index.append(seq) 
df_selected = df[df['kmer_seq'].isin(CG_index)]
df_selected.to_csv(os.path.join(input_path, 'lambda_5mC.CG.csv'), header = None, index = None, sep='\t')
del df_selected, CG_index, df

###5C
df=pd.read_csv(os.path.join(input_path,'lambda_5C.csv'),sep='\t', header = None)
df.columns = names

####Extract 5C
CG_index = list()
for seq in df['kmer_seq']:
    if seq[num_bases:num_bases+2] == 'CG':
        CG_index.append(seq) 
df_selected = df[df['kmer_seq'].isin(CG_index)]
df_selected.to_csv(os.path.join(input_path, 'lambda_5C.CG.csv'), header = None, index = None, sep='\t')
del df_selected, CG_index, df