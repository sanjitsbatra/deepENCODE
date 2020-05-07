#!/usr/bin/python
"""Imputation challenge custom scoring script for all tracks together
Modified for classification
Author:
    Sanjit Singh Batra
"""

import numpy as np
from os.path import join, isfile
import sys
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, pearsonr


def dict_to_array(d, chrs, t):
    """Concat vectors in d
    """
    result = []
    for c in chrs:
        result.extend(d[c][t])
    return np.array(result)


def compute_pearson(yTrue, yPred):
    return np.corrcoef(yTrue, yPred)[0, 1]


def compute_spearman(yTrue, yPred):
    return spearmanr(yTrue, yPred)[0]


NUM_CELL_TYPES = 51
NUM_ASSAY_TYPES = 35

tracks = []


def load_data(data_dir):
    data = {}
    # Load ground TRUTH validation data
    for cell_type in range(NUM_CELL_TYPES):
        for assay_type in range(NUM_ASSAY_TYPES):
            for chrom in [str(k) for k in range(1, 23)] + ['X']:
                fname = 'C{:02}M{:02}.chr{}.npy'.format(cell_type,
                                                        assay_type,
                                                        chrom)
                fname = join(data_dir, fname)
                if isfile(fname):
                    if (True): # (chrom == '7') or (chrom == '4') ):
                        tracks.append((cell_type, assay_type))
                        print('Loading', fname)
                    this_array = np.load(fname)
                    if (cell_type, assay_type) not in data:
                        # TODO: If this isnt't numpy but list, then mem is high!
                        data[(cell_type, assay_type)] = []
                    data[(cell_type, assay_type)].append(
                        this_array
                    )
            if (cell_type, assay_type) in data:
                data[(cell_type, assay_type)] = np.concatenate(
                    data[(cell_type, assay_type)]
                )
    return data


truth_data = load_data(sys.argv[1])
tracks = []
method_data = load_data(sys.argv[2])

# Now, we iterate through each track, concatenate all chromosomes and score
print("Finished loading data")
Final_Result = []
Final_Result.append("Track\tAUROC\tAUPRC\tTop-K\tPearson\tSpearman")

# Reflect the cell type + 1 change into here as well
for track in tracks:
    track_name = 'C{:02}M{:02}'.format(track[0]+0, track[1]+0) 
    # Sanjit made this change on the line above on 9 August 2019

    print("Computing metrics for "+str(track_name))
    yTrue = truth_data[track]  # dict_to_array(truth_data, chroms, track)
    yTrue_binary = np.where(yTrue > 3, 1, 0) # choose threshold from training

    yPred = np.squeeze( method_data[track] ) 
    # Added for ensembling  # dict_to_array(method_data, chroms, track)

    # print("created genome-wide arrays")

    # Compute AUROC  
    auroc_val = round(100.0 * roc_auc_score(yTrue_binary, yPred), 3)
    # print("MSE = "+str(mse_val))

    # Compute AUPRC
    precision_val, recall_val, _ = precision_recall_curve(yTrue_binary,
                                                          yPred)
    auprc_val = round(100.0 * auc(recall_val, precision_val), 3)

    # Compute Top-K accuracy
    K = list(yTrue_binary).count(1)
    top_k = yTrue_binary[np.asarray(yPred).argsort()[(-1*K):][::-1]]
    top_k_val = round(100.0 * ((list(top_k).count(1)*1.0) / (1.0*K)), 3) 
                                        
    pearson_val = compute_pearson(yTrue, yPred)

    spearman_val = compute_spearman(yTrue, yPred)

    Final_Result.append(str(track_name)
                        + "\t" + str(auroc_val)
                        + "\t" + str(auprc_val)
                        + "\t" + str(top_k_val)
                        + "\t" + str(pearson_val)
                        + "\t"+str(spearman_val))

    print(str(Final_Result[-1]))

print(Final_Result)
outfile = open(sys.argv[2]+'/Results.txt', 'w')
outfile.write("\n".join(Final_Result))
outfile.close()
