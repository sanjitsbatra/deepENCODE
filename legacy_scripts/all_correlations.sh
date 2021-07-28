#!/bin/bash

#rm Correlation_Results.txt
#while read i ; do python compute_correlation.py $i >> Correlation_Results.txt ; done < shuffled_unique_correlation_pairs.txt 

rm ct23_chr3.Correlation_Results.txt
while read i ; do python compute_correlation.py $i >> ct23_chr3.Correlation_Results.txt ; done < ct_23_chr3.correlation_pairs.txt
