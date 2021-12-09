#! /usr/bin/env python3
## The script is used to extract raw signal, event, alignment information from single basecalled-resquirred fast5 read
##Usage: python fast5_parser.py --input_path '/pod/2/li-lab/Ziwei/Nanopore/daily/test/000006ea-dddb-429c-8277-a81ce50da6a0.fast5'
import h5py
import numpy as np
import os, sys
import argparse
from statsmodels import robust
from ont_fast5_api.conversion_tools.conversion_utils import get_fast5_file_list
    
class raw_fast5:
    def __init__(self, path, corr_group, basecall_group, basecall_subgroup, signal_group):
        self._fast5_id = os.path.basename(path)
        self._obj = h5py.File(path, mode="r")
        self._corr_group = corr_group
        self._basecall_group = basecall_group
        self._basecall_subgroup = basecall_subgroup
        self._signal_group = signal_group
        
    def __eq__(self, other):
        self._fast5_id == other._fast5_id
    
    def fast5_readid(self):
        f5 = self._obj
        try:
            read_num = list(f5[f'/{signal_group}'])[0]
        except Exception:
            raise RuntimeError('Signal not found.')
                
        read = f5[f'/{signal_group}/{read_num}']
        read_attr = dict(list(read.attrs.items()))
        read_id = read_attr['read_id'].decode('UTF-8') 
        return read_id
    
    def fast5_signal(self):
        f5 = self._obj
        try:
            read_num = list(f5[f'/{signal_group}'])[0]
        except Exception:
            raise RuntimeError('Signal not found.')
            
        signal = f5[f'/{signal_group}/{read_num}/Signal']
        signal_attr = np.array(signal[:])
        return signal_attr
    
    def fast5_event(self):
        try:
            # Get event data after tombo-requirrel
            event = self._obj[f'/Analyses/{self._corr_group}/{self._basecall_subgroup}/Events']
        except Exception:
            raise RuntimeError('Events not found.')
            
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
        return event_attr
    
    def fast5_align(self):
        try:
            strand_align = f5[f'/Analyses/{self._corr_group}/{self._basecall_subgroup}/Alignment']
        except Exception:
            return -1
        
        read_strand = 't' if self._basecall_subgroup=='BaseCalled_template' else 'c'
        
        align_attr = dict(list(strand_align.attrs.items()))
        #Extract chrom, chrom_start, strand for alignment information
        chrom = align_attr['mapped_chrom']
        strand = align_attr['mapped_strand']
        start = align_attr['mapped_start']
        return chrom, strand, start, read_strand
             
#        except Exception:
#            raise RuntimeError('Alignment not found.')
            
    
#def read_fast5(input_path):
#    fast5_fp = get_fast5_file_list(input_path,recursive=True)
#    return fast5_fp

if __name__ == '__main__':
    ## Create the parser #
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
                               'default BaseCalled_template'),
    parser.add_argument('--signal_group', required=False,
                               default='Raw/Reads',
                               help='the signal_group in fast5 files. '
                               'default Raw/Reads')
    
    
    # Execute parse_args()
    args = parser.parse_args()
    corr_group = args.corrected_group
    basecall_group = args.basecall_group
    basecall_subgroup = args.basecall_subgroup
    signal_group=args.signal_group
    
    fast5_fn = args.input_path
    print(fast5_fn)

    f5 = raw_fast5(fast5_fn, corr_group, basecall_group, basecall_subgroup, signal_group)
    event = f5.fast5_event()
    read_id = f5.fast5_readid()
    signal = f5.fast5_signal()
    chrom, strand, start, read_strand = f5.fast5_align()
    chrom, strand, start, read_strand
#    print(test)
    
    print("Testing is done!")

    