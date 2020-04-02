#!/usr/bin/python
"""Imputation challenge custom scoring script for all tracks together
Author:
    Sanjit Singh Batra
"""

import numpy as np
from os.path import join, isfile
import sys
from scipy.stats import spearmanr


def dict_to_array(d, chrs, t):
    """Concat vectors in d
    """
    result = []
    for c in chrs:
        result.extend(d[c][t])
    return np.array(result)


def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2.).mean()


def gwcorr(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]


def gwspear(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]


def mse1obs(y_true, y_pred):
    n = int(y_true.shape[0] * 0.01)
    y_true_sorted = np.sort(y_true)
    y_true_top1 = y_true_sorted[-n]
    idx = y_true >= y_true_top1

    return mse(y_true[idx], y_pred[idx])


def mse1imp(y_true, y_pred):
    n = int(y_true.shape[0] * 0.01)
    y_pred_sorted = np.sort(y_pred)
    y_pred_top1 = y_pred_sorted[-n]
    idx = y_pred >= y_pred_top1

    return mse(y_true[idx], y_pred[idx])


NUM_CELL_TYPES = 51
NUM_ASSAY_TYPES = 35

tracks = []


def load_data(data_dir):
    data = {}
    # Load ground TRUTH validation data
    for cell_type in range(NUM_CELL_TYPES):
        for assay_type in range(NUM_ASSAY_TYPES):
            for chrom in [str(k) for k in range(1, 23)] + ['X']:
                fname = 'C{:02}M{:02}.chr{}.npy'.format(cell_type + 1,
                                                        assay_type + 1,
                                                        chrom)
                fname = join(data_dir, fname)
                if isfile(fname):
                    if ( (chrom == '7') or (chrom == '4') ):
                        tracks.append((cell_type, assay_type))
                        print('Loading', fname)
                    this_array = np.load(fname)
                    if (cell_type, assay_type) not in data:
                        # TODO: If this isnt't numpy but list then mem is high!
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

for track in tracks:
    track_name = 'C{:02}M{:02}'.format(track[0]+0, track[1]+0) # Sanjit made this change on 9 August 2019
    print("Computing metrics for "+str(track_name))
    y_true = truth_data[track]  # dict_to_array(truth_data, chroms, track)
    y_pred = np.squeeze( method_data[track] ) # Added for ensembling  # dict_to_array(method_data, chroms, track)

    # print("created genome-wide arrays")
    mse_val = mse(y_true, y_pred)
    # print("MSE = "+str(mse_val))

    gwcorr_val = gwcorr(y_true, y_pred)
    # print("Pearson = "+str(gwcorr_val))

    gwspear_val = gwspear(y_true, y_pred)
    # print("Spearman = "+str(gwspear_val))

    # print(str(y_true.shape)+"\t"+str(y_pred.shape))
    mse1obs_val = mse1obs(y_true, y_pred)
    mse1imp_val = mse1imp(y_true, y_pred)

    Final_Result.append(str(track_name)
                        + "\t" + str(mse_val)
                        + "\t" + str(gwcorr_val)
                        + "\t" + str(gwspear_val)
                        + "\t" + str(mse1obs_val)
                        + "\t"+str(mse1imp_val))

    print(str(Final_Result[-1]))

print(Final_Result)
outfile = open(sys.argv[2]+'/Results.txt', 'w')
outfile.write("\n".join(Final_Result))
outfile.close()
