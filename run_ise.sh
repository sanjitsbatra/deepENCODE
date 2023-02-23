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
#SBATCH --partition=savio2_htc  # savio3_htc
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
#SBATCH --time=2:30:00
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

transcript=$1

python run_ise.py --transcript ${transcript}

