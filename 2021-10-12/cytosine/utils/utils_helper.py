#! /usr/bin/env python3
## The script is used to
import h5py
import numpy as np
import os
from statsmodels import robust
from ont_fast5_api.conversion_tools.conversion_utils import get_fast5_file_list
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pyfaidx import Fasta
import multiprocessing

##### one hot coding
# https://2-bitbio.com/2018/06/one-hot-encode-dna-sequence-using.html
# https://gist.github.com/rachidelfermi/7fce95fd67e67fa47681e2f7d206c5a3
basepair = {'A': 'T',
            'C': 'G', 
            'G': 'C', 
            'T': 'A'}


class hot_dna:
    def __init__(self, seq):
        seq_array = np.array(list(seq))
        #integer encode the sequence
        seq_integer = LabelEncoder().fit_transform(seq_array) 

        #reshape because that's what OneHotEncoder likes
        seq_integer = seq_integer.reshape(len(seq_integer), 1)
        seq_1hot = OneHotEncoder(sparse=False).fit_transform(seq_integer)
        
        self._seq = seq_array
        self._integer = seq_integer
        self._1hot = seq_1hot
