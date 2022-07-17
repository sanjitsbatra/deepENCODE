#!/bin/bash

#$ -cwd
#$ -N fpUTR
#$ -o ../../Logs/
#$ -e ../../Logs/
#$ -j y
#$ -l h_vmem=30g
#$ -l mem_free=30g

# Move to Code directory and activate the conda environment
cd "/home/sbatra/.chrmt/Code/deepENCODE/"
conda activate gpu_env

inserted_scalar=$1
maximum_inserted_minuslog10_p_value=$2
gaussian_bandwidth=$3
nucleosome_lambda=$4
H3K27me3_flag=$5
dCas9_binding_flag=$6
MNase_scaling_flag=$7
maximum_flag=$8
output_prefix=$9

python run_chrmt_inference.py --inserted_scalar ${inserted_scalar} --maximum_inserted_minuslog10_p_value ${maximum_inserted_minuslog10_p_value} --gaussian_bandwidth ${gaussian_bandwidth} --nucleosome_lambda ${nucleosome_lambda} --H3K27me3_flag ${H3K27me3_flag} --dCas9_binding_flag ${dCas9_binding_flag} --MNase_scaling_flag ${MNase_scaling_flag} --maximum_flag ${maximum_flag} --output_prefix ${output_prefix}

