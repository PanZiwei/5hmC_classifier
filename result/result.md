1. Cytosine site after Megalodon v2.3.4 + Guppy v5.0.14: Category distribution: {0(5C): 5894606, 1(5mC): 9301876, 2(5hmC): 1662}

2. Extract features from single fast5 read after basecalling(Guppy v5.0.11) and resquiggle

Signal feature:
kmer sequence
signal mean for each base in kmer
signal std for each base in kmer
signal number for each base in kmer
signal range for each base in kmer

chrom, site_pos, align_strand, loc_in_ref, read_id, read_strand,
kmer_seq, kmer_signal_mean, kmer_signal_std, kmer_signal_length, kmer_signal_range,

3. pyfaidx:  Start attributes are 1-based, End attributes are 0-based https://github.com/mdshw5/pyfaidx
Note that start and end coordinates of Sequence objects are [1, 0]. This can be changed to [0, 0] by passing one_based_attributes=False to Fasta or Faidx. This argument only affects the Sequence .start/.end attributes, and has no effect on slicing coordinates.

More details: https://github.com/mdshw5/pyfaidx/issues/74#issuecomment-146609127





