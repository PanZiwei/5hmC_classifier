0#!/usr/bin/env python3
"""
Sep2021, Ziwei Pan
The script is for:
Get the list for Guppy-basecalled Tombo-resquirreled fast5 files based on the chromosome information
"""
import pandas as pd
import numpy as np
import sys, os
import h5py
import argparse
from ont_fast5_api.conversion_tools.conversion_utils import get_fast5_file_list
import csv

T4_list = list()
lambda_list = list()
human_list = list()

def get_chrom(input_path, output_path, 
              corr_group='RawGenomeCorrected_001',
              basecall_subgroup='BaseCalled_template'):#, chromOfInterest):
# Collate the attribute list: Opening files(read mode only)
    fast5_fp = get_fast5_file_list(input_path,recursive=True)
    for fast5_fn in fast5_fp:
        f5 = h5py.File(fast5_fn, 'r')
        try:
            strand_align = f5[f'/Analyses/{corr_group}/{basecall_subgroup}/Alignment']
            align_attr = dict(list(strand_align.attrs.items()))
            #Extract chrom, chrom_start, strand for alignment information
            chrom = align_attr['mapped_chrom']
            if chrom == 'KJ477685.1': #T4: KJ477685.1
                T4_list.append(fast5_fn)
            elif chrom == 'J02459.1':
                lambda_list.append(fast5_fn)
            else:
                human_list.append(fast5_fn)
        except Exception:
            pass
#            raise RuntimeError('Alignment not found.')

    T4 = pd.DataFrame(T4_list)
    lambda_ = pd.DataFrame(lambda_list)
    human = pd.DataFrame(human_list)
    
    T4.to_csv(os.path.join(output_path, 'T4_hac_list.tsv'), sep=',',index = None, header=None)
    lambda_.to_csv(os.path.join(output_path, 'lambda_hac_list.tsv'), sep=',',index = None, header=None)
    human.to_csv(os.path.join(output_path, 'human_hac_list.tsv'), sep=',',index = None, header=None)
              
if __name__ == '__main__':
    ## Create the parser #
    parser = argparse.ArgumentParser(description="Extract features from re-squirred fast5 files")
    parser.add_argument('--input_path', required=True, help='input path of fast5 files')
    parser.add_argument('--output_path',required=True, help='output path of fast5 files')
    
    args = parser.parse_args()
    input_path= args.input_path
    output_path = args.output_path
    get_chrom(input_path, output_path)
              
              

    
    
