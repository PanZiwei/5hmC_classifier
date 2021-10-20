#! /usr/bin/env python3
#########I have problem using the multiprocessing(feature_q is empty always), so I didn't use it at this moment
import h5py
import numpy as np
import os, time, multiprocessing
from statsmodels import robust
from pyfaidx import Fasta

from ont_fast5_api.conversion_tools.conversion_utils import get_fast5_file_list

#Import function/class in fast5_parser.py, process_helper.py
from .fast5_helper import fast5_rescale, fast5_normalize_signal, raw_fast5

################ Before extraction
### Get refererence length 
##The same function as `get_contig2len` in DeepSignal, but more straightforward
##
def get_ref_length(ref_path):
    ref_length = dict()
    
    #case-insensitive, and to eliminate a potentially large else-if chain:
    #https://www.biostars.org/p/173963/#338583
    if ref_path.lower().endswith(('.fasta', '.fa')):
        for ref in Fasta(ref_path,"fasta"):
            ref_length[ref.name] = len(ref) #add ref_name:ref_length pair into dictionary
        return ref_length
    else:
        raise TypeError("The reference must be in .fa or .fasta format!")
        
### Define put_file_in_queue
### I borrow the code from `_fill_files_queue` from Deepsignal
def put_file_in_queue(fast5_q, fast5_file, batch_size):
    for i in np.arange(0, len(fast5_file), batch_size):
        fast5_q.put(fast5_file[i:(i+batch_size)])
    return fast5_q

    
############ Get mod location in the reference  
#get_refloc_of_methysite_in_motif in DeepSignal
def get_mod_site_in_ref(seq, motif_seq, mod_loc = 0): #assert the mod_location is 0-based
    """
    Find the motif location in reference in one single fast5 read
    :param seq: read sequence
    :param motif_seq: motif set
    :param mod_loc: 0-based
    :return:
    """
    motif = set(motif_seq)
    seq_length = len(seq)
    motif_length = len(list(motif)[0])
    mod_site_loc = []
    for i in range(0, seq_length - motif_length + 1):
        if seq[i:i + motif_length] in motif:
            mod_site_loc.append(i + mod_loc)
        else:
            continue
    return mod_site_loc

############ Extract perbase feature in single fast5 read
#### Raw signal --> Normalization --> alignment --> methylated site --> features
#Modify from _extract_features in DeepSignal/DeepMP
def extract_perbase_signal_feature(fast5_files, 
                                   corrected_group, basecall_group, basecall_subgroup, signal_group, normalize_method, 
                                   motif_seq, mod_loc, ref_total_length, kmer, mod_label):
    
    feature_list = []
    error_num = 0
    
    for fast5_file in fast5_files: #fast5_files is a list
        try:
            #Get fast5 object
            fast5 = raw_fast5(fast5_file, 
                              corrected_group, basecall_group, basecall_subgroup, signal_group)
            
            # Extract signal
            read_id, fast5_signal = fast5.fast5_signal() #get readid
            #Extract event information: #Raw signal --> Normalization
            event = fast5.fast5_event()
            raw_signal = fast5_rescale(fast5_file, fast5_signal)
            norm_signal = fast5_normalize_signal(raw_signal, normalize_method)
            
            #Assign list to save the features
            basecalled_seq, raw_signal_list, norm_signal_list = "", [], []
            for e in event:
                basecalled_seq += e[2]
                norm_signal_list.append(norm_signal[e[0]:(e[0] + e[1])])  
                assert len(basecalled_seq) == len(norm_signal_list)
                ######Atention!!! The raw signal is not useful at this stage, may need to be stored
#                raw_signal_list.append(raw_signal[e[0]:(e[0] + e[1])])        
#            assert len(norm_signal_list) == len(raw_signal_list)
    
            # Extract other information for the read
            chrom, align_strand, start, read_strand = fast5.fast5_align()
            
            # Get a specific reference genome length
            ref_length = ref_total_length[chrom]
                
            # Correct the direction of the read
            if align_strand == '+':
                chrom_start_in_strand = start
            elif align_strand == '-': #reversed strand
                chrom_start_in_strand = ref_length - (start + len(basecalled_seq)) 
                
            # Locate the motif in the reference
            site_loc = get_mod_site_in_ref(basecalled_seq, motif_seq, mod_loc)
            
            if kmer % 2 != 0:
                num_bases = (kmer - 1) // 2
            else:
                raise ValueError("kmer must be an odd number")

            #Find the motif location in the aligned strand
            for loc_in_read in site_loc:
                if num_bases <= loc_in_read < len(basecalled_seq) - num_bases:
                    loc_in_ref = loc_in_read + chrom_start_in_strand

                    if align_strand == '+':
                        site_pos = loc_in_ref
                    elif align_strand == '-':
                        #Calculate the location in the - strand
                        site_pos = ref_length - 1 - loc_in_ref

                    #Find the sequence for kmer
                    kmer_seq = basecalled_seq[(loc_in_read - num_bases):(loc_in_read + num_bases + 1)]
                    
                    #Find the (normalized) signal information for kmer
                    kmer_signal = norm_signal_list[(loc_in_read - num_bases):(loc_in_read + num_bases + 1)]
                    assert len(kmer_seq) == len(kmer_signal)
                    #Calcuate signal feature for each base in the kmer: mean, std, median, diff
                    kmer_signal_mean = [np.mean(x) for x in kmer_signal]
                    kmer_signal_std = [np.std(x) for x in kmer_signal]
                    kmer_signal_length = [len(x) for x in kmer_signal]
                    kmer_signal_range = [np.abs(np.max(x) - np.min(x)) for x in kmer_signal]
                    
                    
                    ##########I didn't include signal feature at this moment(DeepSignal did)
                    
                    feature_list.append(
                        (chrom, site_pos, align_strand, loc_in_ref, 
                         read_id, read_strand,
                         kmer_seq, kmer_signal_mean, kmer_signal_std, kmer_signal_length, kmer_signal_range,
                         mod_label)
                    )
                    
        except Exception:
            error_num += 1
            continue

    return feature_list, error_num

############ Covert the feature into string type
def feature_to_str(feature):
    #unzip a list of tuples
    chrom, site_pos, align_strand, loc_in_ref, read_id, read_strand, kmer_seq, kmer_signal_mean, kmer_signal_std, kmer_signal_length, kmer_signal_range, mod_label = feature
    
    signal_mean = ','.join([str(x) for x in np.around(kmer_signal_mean, decimals=6)])
    signal_std = ','.join([str(x) for x in np.around(kmer_signal_std, decimals=6)])
    signal_length = ','.join([str(x) for x in np.around(kmer_signal_length, decimals=6)])
    signal_range = ','.join([str(x) for x in np.around(kmer_signal_range, decimals=6)])
    
    return "\t".join([str(chrom), str(site_pos), str(align_strand), str(loc_in_ref), 
                      str(read_id), str(read_strand),
                      str(kmer_seq), signal_mean, signal_std, signal_length, signal_range,
                      str(mod_label)])


############ Modify the code from get_a_batch_features_str in DeepSignal
def get_batch_feature(fast5_path, 
                      corrected_group, basecall_group, basecall_subgroup, signal_group, 
                      normalize_method, motif_seq, mod_loc,ref_total_length, kmer, mod_label):
    #Extract features around CG in each read
    fast5_files = get_fast5_file_list(fast5_path, recursive=True)
    fast5_num = len(fast5_files)

    if fast5_num != 0:
        feature_list, error_num = extract_perbase_signal_feature(fast5_files, 
                                                                 corrected_group, basecall_group, basecall_subgroup, signal_group, 
                                                                 normalize_method, motif_seq, mod_loc, ref_total_length, kmer, mod_label)
            
        feature_str = []
        
        #Covert features into string 
        for feature in feature_list:
            feature_str.append(feature_to_str(feature))
        print("Batch feature extraction is done..")
        return feature_str, error_num

    else:
        print("No fast5 is detected in the input path!")
        return '',''
            
     
            
############ Save features into file, modify from _write_featurestr_to_file in DeepSignal/DeepMP
def save_feature_to_file(output_file, feature_list):
    with open(output_file, 'w') as output_fp:
        for feature in feature_list:
            output_fp.write(feature + "\n")
        output_fp.flush()
                