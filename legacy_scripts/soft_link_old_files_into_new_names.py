# This script takes as input a list of filenames from the old nomenclature (C__M__) 
# to be soft linked to files with the new nomenclature (T__A__)

import sys
from os import path
import subprocess

prefix = "/scratch/sanjit/ENCODE_Imputation_Challenge/2_April_2020/Data/Binned_25bp_ENCODE_Training_and_Validation_Data/"

f = open("../CellTypes_AssayTypes.txt", 'r')
for line in f:
	vec = line.rstrip("\n").split(" ")
	src = vec[0]
	tgt = vec[1]

	# First check if the file exists
	if (path.exists(prefix+src)):			
		# Then soft link it
		# print(src, "soft linking")
		subprocess.Popen(("ln -s "+prefix+src+" "+tgt).split(), stdout=subprocess.PIPE).communicate()
	else:
		print(src, "doesn't exist!")
	
