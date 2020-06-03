# This script takes as input the directory containing all the CT_AT_chr.npy files
# It also takes input a set of experiments to be used as Test data
# It also takes as input a set of chromosomes to work with
# It then creates Training and Testing directories by soft-linking the 
# relevant files into them

import sys, os, numpy as np
import subprocess


# Read in the names of all files
data_path = '/scratch/sanjit/ENCODE_Imputation_Challenge/2_April_2020/Data/Binned_25bp_ENCODE_Training_and_Validation_Data/'

all_data = []
for (dirpath, dirnames, filenames) in os.walk(data_path):
    all_data.extend(filenames)
    break


# Read in the names of the experiments to be used as Train or Test data
current_data = []
f_current = open(sys.argv[1], 'r')
for line in f_current:
	vec = line.rstrip('\n')
	current_data.append(vec)


# Read in the names of the experiments to be used as the Test data
test_data = []
f_test = open(sys.argv[2], 'r')
for line in f_test:
	vec = line.rstrip('\n')
	test_data.append(vec)


# Specify which chromosomes we will be training and testing on
chroms = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr11', 'chr12', 'chr21'] # , 'chr5', 'chr14', 'chr2']


# First we create the Training and Testing directories
subprocess.Popen(("mkdir -p Training_Data").split(), stdout=subprocess.PIPE).communicate()
subprocess.Popen(("mkdir -p Testing_Data").split(), stdout=subprocess.PIPE).communicate()


# Then we populate these directories with soft-links
for file_name in all_data:
	experiment = file_name.split(".")[0]
	chrom = file_name.split(".")[1] 
	if( (chrom not in chroms) or (experiment not in current_data) ):
		continue
	if(experiment in test_data):
		subprocess.Popen(("ln -s "+data_path+file_name+" Testing_Data/"+file_name).split(), stdout=subprocess.PIPE).communicate()
	else:				
		subprocess.Popen(("ln -s "+data_path+file_name+" Training_Data/"+file_name).split(), stdout=subprocess.PIPE).communicate()


print("We have finished creating soft-linked Training and Testing data directories")	


