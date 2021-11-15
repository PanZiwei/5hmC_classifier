0#!/usr/bin/env python3
"""
Sep2021, Ziwei Pan
The script is for:
Seperate the original Guppy-basecalled Tombo-resquirreled fast5 files based on the chromosome information
"""
import h5py
import numpy as np
import os, sys
import subprocess
import glob

def process(fast5_path):#, chromOfInterest):
# Collate the attribute list: Opening files(read mode only)
    hdf = h5py.File(fast5_path, 'r')
# Get the names of all groups and subgroups in the file
    list_of_names = []
    hdf.visit(list_of_names.append)
    attribute = []
    
    chromosome = False
    for name in list_of_names:
    #    Get all the attribute name and value pairs
        itemL = hdf[name].attrs.items()
        for item in itemL:
            attr, val = item
            if attr == "mapped_chrom":
                chromosome = val
    
    return chromosome

def listAndProcessFast5Files(fast5_path, dest_path):
    selectedReads = 0
    discardedReads = 0
    processedFast5 = 0
    presentDirs = {}
    
    for fullFileName in glob.glob(f"{fast5_path}/**/*.fast5",recursive=True):
        file = os.path.basename(fullFileName)
        chromosome = process(fullFileName)
        if chromosome != False:
            if chromosome not in presentDirs:
                if os.path.exists("{}/{}".format(dest_path, chromosome)) == False: #true if path exists 
                    command = "mkdir {}/{}".format(dest_path, chromosome)
                    print(command)
                    subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read()
                    presentDirs[chromosome] = 1

            command = "cp {} {}/{}/{}".format(fullFileName, dest_path, chromosome, file)
            subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read()
            selectedReads += 1
        else:
            discardedReads += 1
        processedFast5 += 1
        if (processedFast5 % 1000) == 0:
            print("processedFast5:", processedFast5)
                    
    print("selectedReads:", selectedReads)
    print("discardedReads:", discardedReads)

fast5_path, dest_path = sys.argv[1:]

listAndProcessFast5Files(fast5_path, dest_path)
