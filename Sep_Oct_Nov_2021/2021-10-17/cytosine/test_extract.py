0#!/usr/bin/env python3
"""
Sep2021, Ziwei Pan
The script is for:
Get the list for Guppy-basecalled Tombo-resquirreled fast5 files based on the chromosome information
"""
import h5py
import numpy as np
import os, sys, time, multiprocessing
from ont_fast5_api.conversion_tools.conversion_utils import get_fast5_file_list

from cytosine.extract_module import extract_fast5_feature

fast5_path='/pod/2/li-lab/Ziwei/Nanopore/daily/test/test'
ref_path = '/pod/2/li-lab/Ziwei/Nanopore/data/reference/hg38.fa'
num_proc = 10

corrected_group='RawGenomeCorrected_001' #tombo resquiggle save location
basecall_group='Basecall_1D_001' #has to be save in the tombo requiggle step
basecall_subgroup='BaseCalled_template' #Attention: the basecall_subgroup can be 'basecalled_compl
signal_group='Raw/Reads'
normalize_method = 'mad'

motif_seq = 'CG'
mod_loc = 0
kmer=17
mod_label=1

output_file='/pod/2/li-lab/Ziwei/Nanopore/daily/2021-10-17/extract_feature_test.tsv'
batch_size = 100

print("Feature extraction begins!")

extract_fast5_feature(fast5_path, ref_path, num_proc, 
                      corrected_group, basecall_group, basecall_subgroup, signal_group, normalize_method,  
                      motif_seq, mod_loc, kmer, mod_label,
                      output_file, batch_size)
print("Feature extraction finished!")
