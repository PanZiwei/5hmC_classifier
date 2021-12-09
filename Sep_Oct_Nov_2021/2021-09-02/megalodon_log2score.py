#! /usr/bin/env python3
"""
Aug 2021, Ziwei Pan
Convert original Megalondon natural log output into regular probablity score


### Example default input format from Megalondon (pre-processed to containing strand-info):
    head /fastscratch/c-panz/K562/megalodon/per_read_modified_base_calls.merged.sorted.bed
    chr1	10468	a0b5e934-3784-4163-b46c-f575ac1015bf	+	-1.6103846163370363	-0.2229070153110119	    m
    chr1	10470	a0b5e934-3784-4163-b46c-f575ac1015bf	+	-2.035311286540776	-0.13999775139928522	m
    chr1	10483	a0b5e934-3784-4163-b46c-f575ac1015bf	+	-1.5477196338381982	-0.2391872270542014 	m
    
Megalodon per-read file is 0-based file. 
col0 - chromosome information
col1 - Start position of CpG
col2 - read_id
col3 - strand information
col4 - 5hmC_prob, probability that the base is hydromethylated/methylated (float expected).
col5 - can_log_prob_col, natural log probability that the base is unmodified (float expected).
col6 - modified state, h-hydromethylated, m-methylated

For a hydroxymethylated base, hydromethylated_prob > methylated_prob;  
For a methylated base, methylated_prob > hydromethylated_prob;
for a unmethylated(canonical) base, can_prob > hydromethylated_prob or can_prob > methylated_prob
"""
import pandas as pd
import numpy as np
import os
import ternary

input_dir='/pod/2/li-lab/Ziwei/Nanopore/results/methyl_call_guppy5.0.11/Megalodon2.3.3'

#5mC_lambda
df=pd.read_csv(os.path.join(input_dir, '5mC_lambda.Megalodon.methyl_call.perRead.bed'),
                names = ['chr','start','read_id','strand','mod_log_prob','can_log_prob','mod_state'],
                sep='\t')
df_convert=df
df_convert['mod_log_prob']=np.e ** (df_convert['mod_log_prob'])
df_convert['can_log_prob']=np.e ** (df_convert['can_log_prob']) 

df_5hmC=df_convert.loc[df_convert['mod_state']=='h']
df_5hmC.columns=['chr','start','read_id','strand','5hmC_prob','5C_prob','mod_state1']

df_5mC=df_convert.loc[df_convert['mod_state']=='m']
df_5mC.columns=['chr','start','read_id','strand','5mC_prob','5C_prob','mod_state2']

df_new=pd.merge(df_5hmC, df_5mC, how="inner", on=['chr','start','read_id','strand','5C_prob'])
df_new=df_new[['chr','start','read_id','strand','5hmC_prob','5mC_prob','5C_prob']]
df_new.to_csv(os.path.join(input_dir, '5mC_lambda.Megalodon.per_read.prob.bed'),header=True, sep='\t', index=None)
del df, df_new, df_convert, df_5hmC, df_5mC

#lambda
df=pd.read_csv(os.path.join(input_dir, 'lambda.Megalodon.methyl_call.perRead.bed'),
                names = ['chr','start','read_id','strand','mod_log_prob','can_log_prob','mod_state'],
                sep='\t')
df_convert=df
df_convert['mod_log_prob']=np.e ** (df_convert['mod_log_prob'])
df_convert['can_log_prob']=np.e ** (df_convert['can_log_prob']) 

df_5hmC=df_convert.loc[df_convert['mod_state']=='h']
df_5hmC.columns=['chr','start','read_id','strand','5hmC_prob','5C_prob','mod_state1']

df_5mC=df_convert.loc[df_convert['mod_state']=='m']
df_5mC.columns=['chr','start','read_id','strand','5mC_prob','5C_prob','mod_state2']

df_new=pd.merge(df_5hmC, df_5mC, how="inner", on=['chr','start','read_id','strand','5C_prob'])
df_new=df_new[['chr','start','read_id','strand','5hmC_prob','5mC_prob','5C_prob']]
df_new.to_csv(os.path.join(input_dir, 'lambda.Megalodon.per_read.prob.bed'),header=True, sep='\t', index=None)
del df, df_new, df_convert, df_5hmC, df_5mC

#T4
df=pd.read_csv(os.path.join(input_dir, 'T4.Megalodon.methyl_call.perRead.bed'),
                names = ['chr','start','read_id','strand','mod_log_prob','can_log_prob','mod_state'],
                sep='\t')
df_convert=df
df_convert['mod_log_prob']=np.e ** (df_convert['mod_log_prob'])
df_convert['can_log_prob']=np.e ** (df_convert['can_log_prob']) 

df_5hmC=df_convert.loc[df_convert['mod_state']=='h']
df_5hmC.columns=['chr','start','read_id','strand','5hmC_prob','5C_prob','mod_state1']

df_5mC=df_convert.loc[df_convert['mod_state']=='m']
df_5mC.columns=['chr','start','read_id','strand','5mC_prob','5C_prob','mod_state2']

df_new=pd.merge(df_5hmC, df_5mC, how="inner", on=['chr','start','read_id','strand','5C_prob'])
df_new=df_new[['chr','start','read_id','strand','5hmC_prob','5mC_prob','5C_prob']]
df_new.to_csv(os.path.join(input_dir, 'T4.Megalodon.per_read.prob.bed'),header=True, sep='\t', index=None)
del df, df_new, df_convert, df_5hmC, df_5mC