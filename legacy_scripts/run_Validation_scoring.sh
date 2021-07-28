#!/bin/bash

while read in; do python /scratch/sanjit/ENCODE_Imputation_Challenge/Scoring/imputation_challenge/score.py /scratch/sanjit/ENCODE_Imputation_Challenge/Validation_Data/"$in".bigwig "$in"".bigwig" --gene-annotations /scratch/sanjit/ENCODE_Imputation_Challenge/Scoring/imputation_challenge/annot/hg38/gencode.v29.genes.gtf.bed.gz --enh-annotations /scratch/sanjit/ENCODE_Imputation_Challenge/Scoring/imputation_challenge/annot/hg38/F5.hg38.enhancers.bed.gz --chrom chr21 --window-size 25 --prom-loc 80 --nth 16 --out "$in"."chr21.score.txt" ; done < ../Validation.indices


