from __future__ import print_function
from __future__ import division
import numpy as np
from os.path import join
from data_loader import BinnedHandlerSeqImputing
from data_loader import NUM_CELL_TYPES, NUM_ASSAY_TYPES
from models import customLoss, maximum_likelihood_loss
from numba import njit
from keras.models import load_model
import sys
import keras_genomics

FPATH = sys.argv[1]

def print_arrays(imputed, validation_indices, chrom):
    for k, idx in enumerate(validation_indices):
        i, j = idx

	# This is being changed because I have changed the cell type + 1 in 
	# the data loader 2 April 2020
        fname = 'C{:02}M{:02}.chr{}'.format(i, j, chrom) 

        outfile = join(FPATH, fname)
        np.save(outfile, np.maximum(imputed[k, :], 0.))


@njit('float32[:, :](float32[:, :, :], int64[:, :])')
def pull_validation(array, indices):
    to_return = np.zeros((indices.shape[0], array.shape[2]),
                         dtype=np.float32)
    # print("pull_validation array.shape=",to_return.shape, array.shape, indices.shape)
    # (15, 51) (51, 35, 901) (15, 2)
    for k in range(indices.shape[0]):
        to_return[k, :] = array[indices[k, 0], indices[k, 1], :]
    return to_return

def process_batch(batch, batch_size, imputed, locus, chrom_len, stride, 
                 CT_exchangeability):
    assert len(batch) <= batch_size
    if len(batch) == 0:
        return locus
    to_skip = 0
    while len(batch) < batch_size:
        batch.append(np.zeros_like(batch[0]))
        to_skip += 1
    batch = np.array(batch)

    batch_output = trained_model.predict(batch) 
    # np.expm1(trained_model.predict(batch)) for regression


    if(CT_exchangeability):
    	batch_output = np.reshape(batch_output,
                              (batch_size, NUM_CELL_TYPES,
                               stride, NUM_ASSAY_TYPES))
        # print(batch_output.shape)
        batch_output = batch_output.transpose((0, 1, 3, 2))
    else:
        batch_output = np.reshape(batch_output,
                              (batch_size, NUM_ASSAY_TYPES,
                               stride, NUM_CELL_TYPES))
        # print(batch_output.shape)
        batch_output = batch_output.transpose((0, 3, 1, 2))

    # print("Batch_output", batch_output.shape)

    for idx, output in enumerate(batch_output):
        if idx >= len(batch) - to_skip:
            break
        if locus + stride > chrom_len:
            diff = chrom_len - locus
        else:
            diff = stride
            imputed[:, locus:locus+diff] = pull_validation(
               output[:, :, :diff], validation_indices)
            locus += stride
    return locus


if __name__ == '__main__':

    # maximum_likelihood_loss(y_true, y_pred, num_output)    
    trained_model = load_model(sys.argv[2],
                                custom_objects={'customLoss': customLoss,
                    'RevCompConv1D': keras_genomics.layers.RevCompConv1D,
                    'RevCompConv1DBatchNorm': keras_genomics.layers.normalization.RevCompConv1DBatchNorm})

    CT_exchangeability = int(sys.argv[3]) > 0

    in_shape = trained_model.inputs[0].shape
    out_shape = trained_model.outputs[0].shape
    num_imputed = NUM_CELL_TYPES * NUM_ASSAY_TYPES

    stride = int(out_shape[1]) // num_imputed
    seg_len = int(in_shape[1]) // (num_imputed + 25 * 4)
    # This was causing a huge difference in the pearson 28 July 2019
    window_size = seg_len - stride + 1  # Changed to  +1?

    batch_size = int(in_shape[0])

    fname = ('/scratch/sanjit/ENCODE_Imputation_Challenge'
             '/2_April_2020/Data/Test.Indices.txt')
    
    f_fname = open(fname, 'r')    

    # Store the indices in a 2D numpy array
    validation_indices = []    
    for line in f_fname:
        cell_t = int(line[1:2+1])
        assay_t = int(line[4:5+1])
        validation_indices.append([cell_t, assay_t])	
    validation_indices = np.asarray(validation_indices)

    # decode each chromosome and save the output as npy binary file
    bwh = BinnedHandlerSeqImputing(window_size, seg_len, CT_exchangeability)
    print('Beginning imputation')
    chrom, _ = bwh.idx_to_chrom_and_start(0)
    chrom_len = bwh.chrom_lens[chrom]
    imputed = np.zeros((validation_indices.shape[0], chrom_len),
                       dtype=np.float32)
    locus = window_size // 2

    batch = []
    idx = 0
    while idx < len(bwh):
        # print("inside loop", np.asarray(batch).shape)
        new_chrom, _ = bwh.idx_to_chrom_and_start(idx)
        if new_chrom != chrom:
            process_batch(batch, batch_size, imputed,
                          locus, chrom_len, stride, CT_exchangeability)
            batch = []
            print('Done with chromosome', chrom)
            print_arrays(imputed, validation_indices, chrom)
            chrom = new_chrom
            chrom_len = bwh.chrom_lens[chrom]
            imputed = np.zeros((validation_indices.shape[0], chrom_len),
                               dtype=np.float32)
            locus = window_size // 2

        batch.append(np.squeeze(bwh[idx]))
        if(idx % 100 == 0):
            print('We have imputed', idx * stride * 25, 'bases')
        idx += 1
        if len(batch) == batch_size:
            locus = process_batch(batch, batch_size, imputed,
                                  locus, chrom_len, stride, CT_exchangeability)
            batch = []
    # Print the final chromosome
    process_batch(batch, batch_size, imputed, locus, chrom_len, stride, CT_exchangeability)
    print_arrays(imputed, validation_indices, chrom)
