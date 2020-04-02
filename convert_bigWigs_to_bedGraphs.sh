#!/bin/bash

fname=${1%.*}

bigWigToBedGraph $1 "/export/home/users/sbatra/Validation_Data_bedgraphs/"${fname}".bedgraph"
