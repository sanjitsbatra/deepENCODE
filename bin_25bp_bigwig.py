import pyBigWig
import sys
import numpy as np
import time

BIN_SIZE = 25

# ALLOWED_CHROMS = ['chr1','chr10','chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19','chr2','chr20', 'chr21', 'chr22','chr3','chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9','chrX']

ALLOWED_CHROMS = ['chr4']

bw = pyBigWig.open(sys.argv[1]+".bigwig")

for chr in ALLOWED_CHROMS:

	start_t = time.clock()

	v = bw.values(chr, 0, bw.chroms(chr), numpy=True)

	# Now we bin the chromosome into BIN_SIZE bp bins
	binned_v = [np.mean(v[i:i+BIN_SIZE]) for i in range(0, len(v), BIN_SIZE)]

	# At the end of the chromosome there are NaNs in the bigwig
	binned_v = np.nan_to_num(binned_v)	

	# Now we write the binned chromosome file for this track:
	np.save(sys.argv[1]+"."+chr, binned_v)

	print "Time taken for binning " + str(chr) + " = " + str(time.clock() - start_t)

