#! /usr/bin/env python3
## The script is used to extract features from multiple fast5 files
##Usage: python extract_module.py --fast5_path $fast5_path --ref_path $ref_path --output_path $output_file
"""
2021/10/17: 
I have problem using the multiprocessing, so I didn't use it at this moment.
Right now: num_proc, batch_size are useless
"""

######batch_size is a useless parameter at this moment
import h5py
import numpy as np
import os, sys
import argparse
import time
from ont_fast5_api.conversion_tools.conversion_utils import get_fast5_file_list

from utils.fast5_helper import *
from utils.extract_helper import *

############################Extract features                
#########I have problem using the multiprocessing, so I didn't use it at this moment
def extract_fast5_feature(fast5_path, ref_path, num_proc, 
                          corrected_group, basecall_group, basecall_subgroup, signal_group, normalize_method,  
                          motif_seq, mod_loc, kmer, mod_label,
                          output_file, batch_size): 
    start_time = time.time()
    
    fast5_files = get_fast5_file_list(fast5_path, recursive=True)
    num_fast5 = len(fast5_files)

    ref_total_length = get_ref_length(ref_path)
    print('Extracting features from nanopore reads...')
    
    feature_str, error_num = get_batch_feature(fast5_path, 
                                               corrected_group, basecall_group, basecall_subgroup, signal_group, 
                                               normalize_method, motif_seq, mod_loc,ref_total_length, kmer, mod_label)
    
    print("Saving features to file...")
    save_feature_to_file(output_file, feature_str)
    
    print("finishing the write_process..")

    print("{}/{} fast5 files failed.\n"
          "extract_feature costs {:.2f} seconds.".format(error_num, num_fast5,
                                                         time.time() - start_time))


    
#    ref_total_length, fast5_q, num_fast5 = extract_preprocess(fast5_path, ref_path, batch_size)  
    #Initiate feature_q, error_q
#    feature_q = Queue()
#    error_q = Queue()

#    print("Before extraction: feature_q is empty:{}".format(feature_q.empty()))
          
    #Extract feature
#    feature_procs = []
    
#    if num_proc > 1:
#        num_proc -= 1

#    for _ in range(num_proc):
#        # get batch feature from single read
#        p = multiprocessing.Process(
#            target=get_batch_feature, 
#            args=(fast5_path, feature_q, error_q,
#                  corrected_group, basecall_group, basecall_subgroup, signal_group, normalize_method,  
#                  motif_seq, mod_loc, ref_total_length, kmer, mod_label)
#        )
#        p.aeomon = True
#        p.start()
#        feature_procs.append(p)
                   
                          
#    print("After extraction: feature_q is empty:{}".format(feature_q.empty()))
#    print("Saving features to file...")
#    # save features to output
#    p_w = multiprocessing.Process(
#        target=save_feature_to_file, 
#        args=(output_file, feature_q)
#        )
#    p_w.daemon = True
#    p_w.start()
        
#    for p in featurestr_procs:
#        p.join()

#    print("finishing the write_process..")

#    p_w.join()
    
#    error_sum = 0
#    while not error_q.empty():
#        error_sum += error_q.get()
#        if not running:
#            break

#    print("{}/{} fast5 files failed.\n"
#          "extract_feature costs {:.2f} seconds.".format(error_sum, num_fast5,
#                                                         time.time() - start_time))
    
    
if __name__ == '__main__':
    ## Create the parser #
    extract_parser = argparse.ArgumentParser(description="Extract features from re-squirred fast5 files")
    ########Input
    extract_input = extract_parser.add_argument_group("Input")
    extract_input.add_argument('--fast5_path', required=True, help='input path of fast5 files')
    extract_input.add_argument('--corrected_group', required=False,
                               default='RawGenomeCorrected_001',
                               help='the corrected_group of fast5 files saved in tombo re-squiggle. default RawGenomeCorrected_001')
    extract_input.add_argument('--basecall_group', required=False,
                               default='Basecall_1D_001',
                               help='the basecall_group of fast5 files after basecalling. default Basecall_1D_001')
    extract_input.add_argument('--basecall_subgroup', required=False,
                               default='BaseCalled_template',
                               help='the basecall_subgroup of fast5 files after basecalling. default BaseCalled_template'),
    extract_input.add_argument('--signal_group', required=False,
                               default='Raw/Reads',
                               help='the signal_group in fast5 files. default Raw/Reads')
    extract_input.add_argument('--ref_path', required=True, help='path for reference genome(.fa or .fasta)')
    
    
    ########Process
    extract = extract_parser.add_argument_group("Process")
    extract.add_argument("--normalize_method", action="store", type=str, 
                         choices=["mad", "zscore"], default="mad", required=False,
                         help="normalizing method(mad or zscore) for signal in read level, default mad")
    extract.add_argument("--motif_seq", action="store", type=str,
                         required=False, default='CG',
                         help='motif seq to be extracted, default: CG')
    extract.add_argument("--mod_loc", action="store", type=int, required=False, default=0,
                         help='0-based location of the targeted base in the motif, default 0')
    extract.add_argument("--mod_label", action="store", type=int,
                         choices=[0,1,2], required=False, default=0,
                         help="the label of the modified cytosine(0-5C, 1-5mC, 2-5hmC),this is for training."
                         "default 0(unmethylated)")
    ##################Attention: more experiments may need to figure out the k-mer!!!!
    extract.add_argument("--kmer", action="store",
                         type=int, required=False, default=17,
                         help="length of kmer. default 17")
    
    
    #####Add remaining argument
    extract_parser.add_argument("--output_path", action="store",
                                type=str, required=True,
                                help='output file path to save the features')
    extract_parser.add_argument("--num_proc", action="store", 
                                type=int, required=False, default=1, 
                                help="number of processes, default 1")
    extract_parser.add_argument("--batch_size", action="store",
                                type=int, required=False, default=100,
                                help="nubmer of files in each process, default 100")
    
    
    #Execute parse_args()
    args = extract_parser.parse_args()
    
    print("Loading input...")
    ##Input
    fast5_path = args.fast5_path
    corrected_group = args.corrected_group
    basecall_group  = args.basecall_group
    basecall_subgroup  = args.basecall_subgroup
    signal_group  = args.signal_group
    ref_path  = args.ref_path
    ##feature
    normalize_method = args.normalize_method
    motif_seq = args.motif_seq
    mod_loc = args.mod_loc
    mod_label = args.mod_label
    kmer = args.kmer
    ##other
    output_file = args.output_path
    num_proc = args.num_proc
    batch_size = args.batch_size
    
    extract_fast5_feature(fast5_path, ref_path, num_proc, 
                          corrected_group, basecall_group, basecall_subgroup, signal_group, normalize_method,  
                          motif_seq, mod_loc, kmer, mod_label,
                          output_file, batch_size)

    
    sys.exit()

 