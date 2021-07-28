# This script takes as input a line from the gene_expression.tsv file and
# computes the cell_type x assay_type input feature matrix for this gene.
# In order to do this, it computes a function based on peaks found "near"
# the TSS of this gene, in each cell_type x assay_type

import sys, os, numpy as np
import pickle as pkl
from pathlib import Path


# This function computes all the peaks in a neighborhood of a tss
def compute_peaks(x, tss):
	W = 2000

	# The input x vector is binned at 25bp
	start = max( int(tss / 25.0) - int(W / 25.0), 0 )

	end = min( int(tss / 25.0) + int(W / 25.0), x.shape[0] )

	peak = 0
	for i in range(start, end+1):
		if(x[i] > 3):
			peak += x[i]
	assert(peak >= 0)
	return peak		


def collect_peaks(data_path, cell_types, assay_types, ct, at, chrom, tss):
	cell_type = cell_types[ct]
	assay_type = assay_types[at]

	file_name = cell_type+assay_type+"."+chrom+".npy"

	# Check if file exists
	if(Path(data_path+"/"+file_name).is_file()):	
		x = np.load(data_path+"/"+file_name)
		peaks = compute_peaks(x, tss)
	else:
		peaks = -1
	return peaks


cell_types = ["C02", "C04", "C10", "C13", "C17", "C20", "C23", "C24", "C29", \
				"C32", "C36", "C46"]

assay_types = ["M02", "M18", "M17", "M16", "M20", "M22", "M29"]

gene_expression_features = {}

data_path = "/scratch/sanjit/ENCODE_Imputation_Challenge/2_April_2020/Data/Binned_25bp_ENCODE_Training_and_Validation_Data"

f = open("/scratch/sanjit/ENCODE_Imputation_Challenge/2_April_2020/Data/Gene_Expression/gene_expression.tsv", 'r')

for line in f:
	vec = line.rstrip("\n").split("\t")
	chrom = vec[0]
	strand = vec[3]
	if(strand == "+"):
		tss = int(vec[1])
	elif(strand == "-"):
		tss = int(vec[2])
	else:
		print("strand is incorrect. Exiting!")
		continue

	transcript_id = vec[4]

	# Now iterate through all the cell_type, assay_type files
	# "Collect" peaks around this tss and 
	# store in cell_type x assay_type matrix
	d = np.full((len(cell_types), len(assay_types)), -1)
	for ct in range(len(cell_types)):
		for at in range(len(assay_types)):			
			peaks = collect_peaks(data_path, cell_types, assay_types, ct, at, chrom, tss)
			if(peaks > 0):
				d[ct][at] = peaks
	
	print("Computed", chrom, transcript_id)
	# Store this input feature matrix as a dictionary, indexed by transcript_id
	gene_expression_features[transcript_id] = d


# Write the gene_expression_features dict to file with a pickle	
with open('gene_expression_features.pkl', 'wb') as output:
	pickle.dump(gene_expression_features, output, protocol=pickle.HIGHEST_PROTOCOL)



