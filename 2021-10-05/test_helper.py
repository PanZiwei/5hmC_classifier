#! /usr/bin/env python3
##python example.py --input_path '/pod/2/li-lab/Ziwei/Nanopore/daily/test/'
##
import h5py
import numpy as np
import os, sys
import subprocess
import glob
import argparse

class raw_fast5:
    def __init__(self, path, corr_group, basecall_group, basecall_subgroup):
        self._path = path
        self._read_id = os.path.basename(path) #return the readid
    
    def __eq__(self, other):
        self._read_id == other._read_id
    
if __name__ == '__main__':
    ## Create the parser #
    input_path='/pod/2/li-lab/Ziwei/Nanopore/daily/test/'
    parser = argparse.ArgumentParser(description="Extract features from re-squirred fast5 files")
    parser.add_argument('--input_path', required=True, help='input path of fast5 files')
    parser.add_argument('--corrected_group', required=False,
                               default='RawGenomeCorrected_001',
                               help='the corrected_group of fast5 files saved in tombo re-squiggle.'
                               'default RawGenomeCorrected_001')
    parser.add_argument('--basecall_group', required=False,
                               default='Basecall_1D_001',
                               help='the basecall_group of fast5 files after basecalling. '
                               'default Basecall_1D_001')
    parser.add_argument('--basecall_subgroup', required=False,
                               default='BaseCalled_template',
                               help='the basecall_subgroup of fast5 files after basecalling. '
                               'default BaseCalled_template')
    
    
    # Execute parse_args()
    args = parser.parse_args()
    print(vars(args))
#    corrected_group = args.corrected_group
#    basecall_subgroup = args.basecall_subgroup
    