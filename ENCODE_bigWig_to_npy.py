import pyBigWig
import sys
import numpy as np
import time


ALLOWED_CHROMS = ["chr"+str(x) for x in range(1, 23)] + ["chrX"]
BIN_SIZE = 25

bw = pyBigWig.open(sys.argv[1])
output_prefix = sys.argv[2]

for chrom in ALLOWED_CHROMS:

    start_t = time.clock()
    v = bw.values(chrom, 0, bw.chroms(chrom), numpy=True)

    # Now we bin the chromosome into BIN_SIZE bp bins
    binned_v = [np.mean(v[i:i+BIN_SIZE]) for i in range(0, len(v), BIN_SIZE)]

    # At the end of the chromosome there are NaNs in the bigwig
    binned_v = np.nan_to_num(binned_v)

    # Bin the numpy array into 100bp instead of 25bp
    temp_output = np.asarray([0] * (4 - binned_v.shape[0] % 4))
    final_npy = np.concatenate([binned_v, temp_output]).reshape(-1, 4)
    final_npy = np.mean(final_npy, axis=1)

    np.save(output_prefix+"."+chrom+".npy", final_npy)

    print("Time taken for binning " + str(chrom) + " " +
          str(time.clock() - start_t), file=sys.stderr)
