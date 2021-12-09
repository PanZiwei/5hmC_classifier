#! /usr/bin/env python3
## The script is used to extract raw signal, event, alignment information from single basecalled-resquirred fast5 read
##Usage: python fast5_parser.py --input_path '/pod/2/li-lab/Ziwei/Nanopore/daily/test/000006ea-dddb-429c-8277-a81ce50da6a0.fast5'
import h5py
import numpy as np
import os, sys
import argparse
import statsmodels as sm
import statsmodels.robust
from ont_fast5_api.conversion_tools.conversion_utils import get_fast5_file_list
    
class raw_fast5:
    def __init__(self, path, corr_group, basecall_group, basecall_subgroup, signal_group):
        self._path = path
        self._fast5_id = os.path.basename(path)
        self._corr_group = corr_group
        self._basecall_group = basecall_group
        self._basecall_subgroup = basecall_subgroup
        self._signal_group = signal_group
        
    def __eq__(self, other):
        self._fast5_id == other._fast5_id
    
    def fast5_signal(self):
        try:
            f5 = h5py.File(self._path, mode="r")
        except IOError:
            raise IOError('Error opening file')

        if f'/{self._signal_group}' in f5:
            read_num = list(f5[f'/{self._signal_group}'])[0]
                
            read = f5[f'/{self._signal_group}/{read_num}']
            read_attr = dict(list(read.attrs.items()))
            read_id = read_attr['read_id'].decode('UTF-8') 
                
            signal = f5[f'/{self._signal_group}/{read_num}/Signal']
            signal_attr = np.array(signal[:])
            
        else:
            print("File {} didn't have read info".format(self._fast5_id))
            read_id = ''
            signal_attr = ''
        
        return read_id, signal_attr

    def fast5_event(self):
        try:
            f5 = h5py.File(self._path, mode="r")
        except IOError:
            raise IOError('Error opening file')
            
        if f'/Analyses/{self._corr_group}/{self._basecall_subgroup}/Events' in f5:
            # Get event data after tombo-requirrel
            event = f5[f'/Analyses/{self._corr_group}/{self._basecall_subgroup}/Events']
            corr_attr = dict(list(event.attrs.items()))
        
            #get read location relatively to the reference genome
            read_start_rel_to_raw = corr_attr['read_start_rel_to_raw']   
        
            #Calculate the start position relatively to the reference genome
            start = list(map(lambda x: x + read_start_rel_to_raw, event['start']))
            #Get the event length for each base
            length = event['length'].astype(np.int)
            #Get the seq information
            seq = [x.decode("UTF-8") for x in event['base']]
        
            assert len(event) == len(start) == len(seq) == len(length)
            event_attr = list(zip(start, length, seq))

        else:
            print("File {} didn't have event info\n".format(self._fast5_id))
            event_attr = ''
            
        return event_attr
        
    def fast5_align(self):
        try:
            f5 = h5py.File(self._path, mode="r")
        except IOError:
            raise IOError('Error opening file')
            
        if f'/Analyses/{self._corr_group}/{self._basecall_subgroup}/Alignment' in f5:
            strand_align = f5[f'/Analyses/{self._corr_group}/{self._basecall_subgroup}/Alignment']
        
            read_strand = 't' if self._basecall_subgroup=='BaseCalled_template' else 'c'
            align_attr = dict(list(strand_align.attrs.items()))
            #Extract chrom, chrom_start, strand for alignment information
            chrom = align_attr['mapped_chrom']
            strand = align_attr['mapped_strand']
            start = align_attr['mapped_start']
        else:
            print("File {} didn't have alignment info".format(self._fast5_id))
            chrom, strand, start, read_strand = '', '', '', '', ''
        
        return chrom, strand, start, read_strand
    
    
######### Signal rescalling
#current in pA = (signal_value + channels 0pA adc) * digitisable range in pA / digitisation
#https://github.com/grimbough/IONiseR/blob/47d8ab1e1d798f3591407be679076a1a5b5d9dd2/R/squiggle.R#L81

#channels 0pA adc: /UniqueGlobalKey/channel_id/{offset}
#digitisable range in pA: /UniqueGlobalKey/channel_id/{range}
#digitisation: /UniqueGlobalKey/channel_id/{digitisation}

######signal_rescale (back to original signal)
def fast5_rescale(path, fast5_signal, channel_group='UniqueGlobalKey/channel_id'):
    try:
        f5 = h5py.File(path, mode="r")
    except IOError:
        raise IOError('Error opening file')
    
    channel = f5[f'/{channel_group}']
    channel_attr = dict(list(channel.attrs.items()))
    raw_signal = np.array((fast5_signal + channel_attr['offset']) * channel_attr['range'] / channel_attr['digitisation']) 
    return raw_signal

######signal normalization with mad method
def fast5_normalize_signal(signal, method='mad'):
    if signal is not None:
        if method == 'mad':
            #assuming a normal distribution 
            #https://stackoverflow.com/a/40619633
            shift, scale = np.median(signal), np.float(sm.robust.mad(signal))
        elif method == 'zscore':
            shift, scale = np.mean(signal), np.float(np.std(signal))
        else:
            raise ValueError('Normalize method is not recogized. Should be mad or zscore')
            
        #signal normalization: https://nanoporetech.github.io/tombo/resquiggle.html#tombo-fast5-format
        #NormSignal=(RawSignal−Shift)/Scale
        ##Attention: can't import statsmodels directly: https://www.py4u.net/discuss/169011
        # There may be problem after tombo 1.3, see explanation above
        norm_signal = (signal - shift) / scale #POTENTIAL ISSUE!
        assert len(signal) == len(norm_signal)
        norm_signal = np.around(norm_signal, 6)
        return norm_signal
    else:
        return ''

    