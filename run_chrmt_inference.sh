#!/bin/bash

# Job name:
#SBATCH --job-name=chrmt_inference
#
# Account:
#SBATCH --account=fc_songlab  # co_songlab
#
# QoS:
#SBATCH --qos=savio_normal  # songlab_htc3_normal
#
# Partition:
#SBATCH --partition=savio3 # savio2_htc  # savio3_htc
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks needed for use case (example):
#SBATCH --ntasks=1
#
# Processors per task:
#SBATCH --cpus-per-task=1
#
# Wall clock limit:
#SBATCH --time=3:30:00
#
# Memory required:
#SBATCH --mem=20G


module load gnu-parallel/2019.03.22
module load java

chrmt_Code_Folder="/global/scratch/projects/fc_songlab/sanjitsbatra/Epigenome_Editing/USFC_chrmt_folder/chrmt/Code/deepENCODE"

# Move to Code directory and activate the conda environment
cd ${chrmt_Code_Folder} 
conda init bash
conda activate gpu_env

model_type=$1
inserted_peak_width=$2
ise_radius=$3
stranded_MNase_offset=$4
inserted_scalar=$5
maximum_inserted_minuslog10_p_value=$6
gaussian_bandwidth=$7
nucleosome_lambda=$8
H3K27me3_flag=$9
dCas9_binding_flag=${10}
MNase_scaling_flag=${11}
maximum_flag=${12}
output_prefix=${13}

python run_chrmt_inference.py --model_type ${model_type} --inserted_peak_width ${inserted_peak_width} --ise_radius ${ise_radius} --stranded_MNase_offset ${stranded_MNase_offset} --inserted_scalar ${inserted_scalar} --maximum_inserted_minuslog10_p_value ${maximum_inserted_minuslog10_p_value} --gaussian_bandwidth ${gaussian_bandwidth} --nucleosome_lambda ${nucleosome_lambda} --H3K27me3_flag ${H3K27me3_flag} --dCas9_binding_flag ${dCas9_binding_flag} --MNase_scaling_flag ${MNase_scaling_flag} --maximum_flag ${maximum_flag} --output_prefix ${output_prefix}

