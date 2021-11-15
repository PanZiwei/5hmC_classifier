Convert DNA sequence/parsing Phred quality with tensorflow2
```python
import tensorflow_io as tfio
import tensorflow as tf

## Download some sample data:
!curl -OL https://raw.githubusercontent.com/tensorflow/io/master/tests/test_genome/test.fastq

#One hot encodings
one_hot = tfio.genome.sequences_to_onehot(fastq_data.sequences)

# Read FASTQ data
fastq_data = tfio.genome.read_fastq(filename="test.fastq")
print(fastq_data.sequences)
print(fastq_data.raw_quality)

tf.Tensor(
[b'GATTACA'
 b'CGTTAGCGCAGGGGGCATCTTCACACTGGTGACAGGTAACCGCCGTAGTAAAGGTTCCGCCTTTCACT'
 b'CGGCTGGTCAGGCTGACATCGCCGCCGGCCTGCAGCGAGCCGCTGC' b'CGG'], shape=(4,), dtype=string)
tf.Tensor(
[b'BB>B@FA'
 b'AAAAABF@BBBDGGGG?FFGFGHBFBFBFABBBHGGGFHHCEFGGGGG?FGFFHEDG3EFGGGHEGHG'
 b'FAFAF;F/9;.:/;999B/9A.DFFF;-->.AAB/FC;9-@-=;=.' b'FAD'], shape=(4,), dtype=string)


###Quality
quality = tfio.genome.phred_sequences_to_probability(fastq_data.raw_quality)
print(quality.shape)
print(quality.row_lengths().numpy())
print(quality)

###One hot encoding
one_hot = tfio.genome.sequences_to_onehot(fastq_data.sequences)
print(one_hot)
<tf.RaggedTensor [[[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]], [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]]]>
```

Each nucleotide in each sequence is mapped as follows: A -> [1, 0, 0, 0] C -> [0, 1, 0, 0] G -> [0 ,0 ,1, 0] T -> [0, 0, 0, 1]

Reference: https://www.tensorflow.org/io/tutorials/genome?hl=hr

tfio.genome.sequences_to_onehot: https://www.tensorflow.org/io/api_docs/python/tfio/genome/sequences_to_onehot#used-in-the-notebooks