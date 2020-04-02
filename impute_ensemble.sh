#!/bin/bash

# We iterate through every track and every chromosome and output the ensembled track as a npy file
# USAGE: bash impute_ensemble.sh <path_to_model.hdf5> <output_dir/>
# while read in; do for i in $(seq 7 7) ; do python2.7 /scratch/sanjit/ENCODE_Imputation_Challenge/ensembling/infer_ensemble.py "$in" "chr"$i $1 $2 ;done ;done < /scratch/sanjit/ENCODE_Imputation_Challenge/ensembling/VIndices
chr_name=$1
while read in;do python2.7 infer_ensemble.py "$in" "chr"${chr_name} final_ensemble_model.hdf5 Final_Ensembling_log1p_plus_2/ ;done < Blind_Tracks
