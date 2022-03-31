#!/bin/bash

# Following the logic in this paper: https://doi.org/10.7554/eLife.16970
# we follow the steps described in https://github.com/airoldilab/cplate
# to process the fastq files into SAM then sorted BAM
# then convert reads into counts of centers per base pair
# this is done as: center ~ start + tlen // 2
# we use scripts from https://github.com/awblocker/paired-end-pipeline for this

# Step - 1: align reads to hg38
# time bowtie --chunkmbs 2000 --phred33-quals -q -v 2 -n 2 --best -M 1 -I 10 -X 3000 --threads 8 -S GCA_000001405.15_GRCh38_no_alt_analysis_set -1 ${run_name}_R1_001.fastq.gz -2 ${run_name}_R2_001.fastq.gz 1>${run_name}.sam 2>${run_name}.log

# Since the scripts from within paired-end-pipeline no longer work
# we use their essential idea of computing the center of the min(start) and min(end)
# across the two reads within a read pair, to be assigned as the read pair's location
# To do this, we first align with bwa mem instead of bowtie since the latter had a small fraction of reads aligning

run_name=$1
mkdir ${run_name}.tmp

# Step - 1: align reads to hg38 with bwa mem
# we also preserve only properly paired reads and convert to sorted BAM file
bwa mem -t 16 hg38/hg38.GCA_000001405.15.fasta ${run_name}_R1_001.fastq.gz ${run_name}_R2_001.fastq.gz | samtools view -b -f 2 -F 524 -F 2048 -@ 16 - | samtools sort -T ${run_name}.tmp -@ 16 - > ${run_name}.sorted.bam
samtools index -@ 16 ${run_name}.sorted.bam

# Step - 2: compute start and end positions of each read
bedtools bamtobed -i ${run_name}.sorted.bam | sort -k 1,1V -k 4,4V -k 3,3g -k 4,4g > ${run_name}.bed

# Step - 3: for each read pair compute its midpoint and generate a bed file containing number of read pairs having a midpoint at each position as the value
python3 generate_MNase_midpoints.py ${run_name}.bed ${run_name}

# Step - 4: for each read pair generate a bed file containing the start and end of the insert with a 1 in the fourth column
python3 generate_MNase_coverage.py ${run_name}.bed ${run_name}

# Step - 5: After running these steps for all three biological replicates, we concatenate across all replicates and sort the resulting bed file
cat Biological-Rep1.MNase_coverage.bedGraph Biological-Rep2.MNase_cov
erage.bedGraph Biological-Rep3.MNase_coverage.bedGraph | sort -k1,1 -k2,2n - > HEK293T.MNase.coverage.hg38.sorted.bed

# Step - 6: Compute genome-wide coverage from the resulting sorted bed file
bedtools genomecov -bg -i HEK293T.MNase.coverage.hg38.sorted.bed -g hg38/h
g38.chrom.sizes > HEK293T.MNase.coverage.hg38.sorted.bedGraph

# Note that all of this might potentially be equivalent to bedtools genomecov -pc on the sorted, merged BAM file
