#!/usr/bin/python2.7

import sys
import numpy as np
import pyBigWig

# We are using hg38 for the ENCODE Imputation Challenge

chrom_list = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX"]

chrom_len = {}
chrom_len["chr1"] = 248956422
chrom_len["chr2"] = 242193529
chrom_len["chr3"] = 198295559
chrom_len["chr4"] = 190214555
chrom_len["chr5"] = 181538259
chrom_len["chr6"] = 170805979
chrom_len["chr7"] = 159345973
chrom_len["chr8"] = 145138636
chrom_len["chr9"] = 138394717
chrom_len["chr10"] = 133797422
chrom_len["chr11"] = 135086622
chrom_len["chr12"] = 133275309
chrom_len["chr13"] = 114364328
chrom_len["chr14"] = 107043718
chrom_len["chr15"] = 101991189
chrom_len["chr16"] = 90338345
chrom_len["chr17"] = 83257441
chrom_len["chr18"] = 80373285
chrom_len["chr19"] = 58617616
chrom_len["chr20"] = 64444167
chrom_len["chr21"] = 46709983
chrom_len["chr22"] = 50818468
chrom_len["chrX"] = 156040895

name = sys.argv[1]
C = name[1:3]
M = name[4:6]

# Load all chromosomes at 25bp resolution for this track
track = {}
for chr in chrom_list:
	track[chr] = np.load(''.join(["C",C,"M",M,".",chr,".npy"]))

# Prepare bigwig to output this track
bw = pyBigWig.open(''.join(["C",C,"M",M,".bigwig"]), "w")
bw.addHeader([("chr1", 248956422), ("chr2", 242193529), ("chr3", 198295559), ("chr4", 190214555), ("chr5", 181538259), ("chr6", 170805979), ("chr7", 159345973), ("chr8", 145138636), ("chr9", 138394717), ("chr10", 133797422), ("chr11", 135086622), ("chr12", 133275309), ("chr13", 114364328), ("chr14", 107043718), ("chr15", 101991189), ("chr16", 90338345), ("chr17", 83257441), ("chr18", 80373285), ("chr19", 58617616), ("chr20", 64444167), ("chr21", 46709983), ("chr22", 50818468), ("chrX", 156040895)])

for chr in chrom_list:
	t = track[chr]
	e = range(25,25*len(t),25)
	e.append(chrom_len[chr])
	print("Now writing "+name+" "+chr)

	bw.addEntries([chr]*len(t), range(0,25*len(t),25), ends=e, values=t)

bw.close()

