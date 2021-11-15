#! /usr/bin/env python3
"""
Aug 2021, Ziwei Pan
Convert original Megalondon natural log output into regular probablity score
"""
import pandas as pd
import numpy as np
import os, sys, argparse

def log2score_strand(input_file, output_file):
    df=pd.read_csv(str(input_file),
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
    
    #Correct negative strand
    df_new.loc[df_new.strand=='-', 'start'] = df_new['start'] + 1
    df_new.to_csv(str(output_file), header=True, sep='\t', index=None)
    del df, df_new, df_convert, df_5hmC, df_5mC

def main():
    parser = argparse.ArgumentParser(description='Convert original Megalondon natural log output into regular probablity score')
    parser.add_argument('--input_path', '-i', action="store", type=str, required=True,
                        help='a directory contains the merged result after megalodon methylation calling.')
    parser.add_argument('--output_path', '-o', action="store", type=str, required=True,
                        help='the file path to save the result')
    args = parser.parse_args()

    input_file = os.path.abspath(args.input_path)
    output_file = os.path.abspath(args.output_path)
    
    if os.path.isfile(input_file):
#        print(input_file, output_file)
        log2score_strand(input_file, output_file)
    else:
        raise TypeError()
        print("The input and output should be file name")

if __name__ == '__main__':
    sys.exit(main())
    