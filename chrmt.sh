#!/bin/bash

# This script trains a chrmt model using ENCODE epigenetic data to predict gene expression
# It also performs perturbation inference on CRISPRa data from the Hilton Lab
# It then computes correlations across cell types and within cell types across genes
# It then creates a unified report for the model

#$ -q gpu
#$ -l gpus=1
#$ -cwd
#$ -N fpUTR
#$ -o ../../Logs/
#$ -e ../../Logs/
#$ -j y
#$ -l h_vmem=60g
#$ -l mem_free=60g

# Move to Code directory and activate the conda environment
cd "/home/sbatra/.chrmt/Code/deepENCODE/"
conda activate gpu_env

output_prefix=$1
window_size=$2
num_layers=$3
num_filters=$4

model_type=$5
loss=$6
mle_lambda=$7

cell_type_index=$8

run_name=${output_prefix}"_0_transcriptome_"${window_size}"_"${num_layers}"_"${num_filters}"_"${model_type}"_"${loss}"_"${mle_lambda}"_"${cell_type_index}

# Step - 1: Train a model
python chrmt_train.py --run_name ${run_name} --framework transcriptome --window_size ${window_size} --num_layers ${num_layers} --num_filters ${num_filters} --model ${model_type} --loss ${loss} --mle_lambda ${mle_lambda} --cell_type_index ${cell_type_index}

# Step - 2: Perform inference on the CRISPRa data from the Hilton Lab
# python chrmt_inference.py --run_name ${run_name} --trained_model ../../Models/${run_name}.hdf5 --window_size ${window_size}
