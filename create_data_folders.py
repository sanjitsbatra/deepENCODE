# This script takes as input the directory containing all the CT_AT_chr.npy files
# It also takes as input a set of train and test chromosomes to work with
# It then creates Training and Testing directories by soft-linking the 
# relevant files into them

import sys, os, numpy as np
import subprocess


# Read in the names of all files
data_path = '/scratch/sanjit/ENCODE_Imputation_Challenge/2_April_2020/Data/NewNomenclature_Binned_25bp_ENCODE_Training_and_Validation_Data/'

all_data = []
for (dirpath, dirnames, filenames) in os.walk(data_path):
    all_data.extend(filenames)
    break


# Specify which chromosomes we will be training and testing on
# Used for training
train_chroms = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr11', 'chr12', 'chr21']

# Used for testing
test_chroms = ['chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr13', 'chr14', 'chr22']

# First we create the Training and Testing directories
subprocess.Popen(("mkdir -p Training_Data").split(), stdout=subprocess.PIPE).communicate()
subprocess.Popen(("mkdir -p Testing_Data").split(), stdout=subprocess.PIPE).communicate()


# Then we populate these directories with soft-links
for file_name in all_data:
	experiment = file_name.split(".")[0]
	chrom = file_name.split(".")[1] 
	if( ((chrom in train_chroms) or
            (chrom in test_chroms)) ): 

	    if(chrom in train_chroms):
		    subprocess.Popen(("ln -s "+data_path+file_name+" Training_Data/"+file_name).split(), stdout=subprocess.PIPE).communicate()
	    else:				
		    subprocess.Popen(("ln -s "+data_path+file_name+" Testing_Data/"+file_name).split(), stdout=subprocess.PIPE).communicate()


print("We have finished creating soft-linked Training and Testing data directories")	


