#!/bin/bash

#USAGE: bash impute_and_score.sh <directory_to_impute_in> <path_to_model_hdf5>
mkdir -p $1
python2.7 /scratch/sanjit/ENCODE_Imputation_Challenge/Smaller_Data/ENCODE_NN/impute.py $1 $2
python2.7 /scratch/sanjit/ENCODE_Imputation_Challenge/Smaller_Data/ENCODE_NN/score_together.py /scratch/sanjit/ENCODE_Imputation_Challenge/Smaller_Data/chr7_Validation $1
