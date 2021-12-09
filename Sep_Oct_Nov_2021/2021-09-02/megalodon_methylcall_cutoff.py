#! /usr/bin/env python3
"""
Aug 2021, Ziwei Pan
calculate methylation frequency at per-site level for Megalodon from per-read level

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
col4 - 5hmC_prob, probability that the base is hydromethylated (float expected).
col5 - 5mC_prob, probability that the base is methylated (float expected).
col6 - 5C_prob, probability that the base is unmodified (float expected).

2-way classifier: 5hmC/non_5hmC or 5mC/non_5mC
if P(5hmC) > cutoff: 5hmC base; P(5hmC) < 1-cutoff: non_5hmC base(non_5hmC = 5mC + 5C)
if P(5mC) > cutoff: 5mC base; P(5mC) < 1-cutoff: non_5mC base(non_5mC = 5hmC + 5C)
"""
import os, sys, gzip, argparse
import pandas as pd
import numpy as np


class SiteStats:
    def __init__(self, chrom, strand, start):
        self._chrom = chrom
        self._strand = strand
        self._start = start
        
        self.hydroxy_coverage = 0
        self.methyl_coverage = 0
        self.hydroxymethyl = 0
        self.unhydroxymethyl = 0
        self.methyl = 0
        self.unmethyl = 0

def calculate_methyl_freq(input_files):
    cpgDict = dict()  
    for i in input_files:
        with open(i, 'r') as f_in:
            next(f_in) #skip header
#        with gzip.open(i, 'rb') as f_in:        
            for line in f_in:
                record = line.strip().split("\t")
                chrom = record[0]
                start = int(record[1])
                strand=record[3]
                hydroxymethyl_prob = float(record[4])
                methyl_prob = float(record[5]) 
                unmethyl_prob=float(record[6])

                if strand == "-":
                    start = start + 1
                elif strand == "+":
                    start = start
                else:
                    raise Exception(f'The file [{input_files}] can not recognized strand-info from row={i}, please check it')
                
                site_key = (chrom, start, strand)
                if site_key not in cpgDict:
                    cpgDict[site_key] = SiteStats(chrom, strand, start)
                                    
                if hydroxymethyl_prob > args.number_of_threshold: ##Count hydroxymethylated reads
                    cpgDict[site_key].hydroxymethyl += 1
                    cpgDict[site_key].hydroxy_coverage += 1
                elif hydroxymethyl_prob < 1 - args.number_of_threshold:  ##Count unmethylated reads
                    cpgDict[site_key].unhydroxymethyl += 1
                    cpgDict[site_key].hydroxy_coverage += 1
                else:  ## Neglect other cases
                    continue
                
                if methyl_prob > args.number_of_threshold: ##Count hydroxymethylated reads
                    cpgDict[site_key].methyl += 1
                    cpgDict[site_key].methyl_coverage += 1
                elif methyl_prob < 1 - args.number_of_threshold:  ##Count unmethylated reads
                    cpgDict[site_key].unmethyl += 1
                    cpgDict[site_key].methyl_coverage += 1
                else:  ## Neglect other cases
                    continue
    return cpgDict

def write_SiteStats(cpgDict, output_file):
    sorted_keys = sorted(list(cpgDict.keys()), key = lambda x: x)
    with open(output_file, 'w') as result:
        result.write('\t'.join(["chrom", "start", "end", "strand", 
                                "5hmC_cov", "non_5hmC_cov","5hmC_coverage", "5hmC_freq", "non_5hmC_freq",
                                "5mC_cov", "non_5mC_cov","5mC_coverage", "5mC_freq", "non_5mC_freq",]) + '\n')
        for key in sorted_keys: 
            cpg_state = cpgDict[key]
            assert(cpg_state.hydroxy_coverage == (cpg_state.hydroxymethyl + cpg_state.unhydroxymethyl))
            assert(cpg_state.methyl_coverage == (cpg_state.methyl + cpg_state.unmethyl))
            if cpg_state.hydroxy_coverage > 0 or cpg_state.methyl_coverage > 0:
                f_hydroxymethyl = float(cpg_state.hydroxymethyl) / cpg_state.hydroxy_coverage
                f_unhydroxymethyl = 1 - f_hydroxymethyl
                
                f_methyl = float(cpg_state.methyl) / cpg_state.methyl_coverage
                f_unmethyl = 1 - f_methyl

                result.write("%s\t%s\t%s\t%s\t%d\t%d\t%d\t%.3f\t%.3f\t%d\t%d\t%d\t%.3f\t%.3f\n" % (cpg_state._chrom, cpg_state._start, cpg_state._start + 1, cpg_state._strand, 
                                                                                                   cpg_state.hydroxymethyl, cpg_state.unhydroxymethyl,cpg_state.hydroxy_coverage,f_hydroxymethyl, f_unhydroxymethyl,
                                                                                                   cpg_state.methyl, cpg_state.unmethyl,cpg_state.methyl_coverage,f_methyl, f_unmethyl))
            else:
                    print("No coverage is detected")

                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate hydroxymethylation frequency at genomic CpG sites for Megalodon results')
    parser.add_argument("-i", "--input_path", action="store", type=str, 
                        help='input file path', required=True)
    parser.add_argument('-o', '--output_path', action="store", type=str,
                        help='output file path', required=True)
    parser.add_argument("-n", "--number_of_threshold", default=0.0, type=float, 
                        help='Threshold for methylation calling', required=False)
    args = parser.parse_args()
    
    input_path = os.path.abspath(args.input_path)
    output_file = args.output_path

    input_files = []
    if os.path.isdir(input_path):
        for file in os.listdir(input_path):
            input_files.append('/'.join([input_path, file]))
    elif os.path.isfile(input_path):
        input_files.append(input_path)
    else:
        raise ValueError()
    print("get {} input file(s)..".format(len(input_files)))
    
    cpgDict = calculate_methyl_freq(input_files)
    write_SiteStats(cpgDict, output_file)