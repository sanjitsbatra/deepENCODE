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


if __name__ == '__main__':

    # maximum_likelihood_loss(y_true, y_pred, num_output)    
    trained_model = load_model(sys.argv[2],
                                custom_objects={'customLoss': customLoss,
                    'RevCompConv1D': keras_genomics.layers.RevCompConv1D,
                    'RevCompConv1DBatchNorm': keras_genomics.layers.normalization.RevCompConv1DBatchNorm})

    CT_exchangeability = int(sys.argv[3]) > 0

    in_shape = trained_model.inputs[0].shape

    window_size = int( int(in_shape[2]) / 2 ) 
    batch_size = int(in_shape[0])

    # Don't dropout any epigenetic tracks at prediction time
    bwh = BinnedHandlerSeqPredicting(window_size, batch_size, drop_prob = 0, CT_exchangeability)
    print('Beginning prediction')

    yTest = []
    yPred = []

    batch = []
    idx = 0
    while idx < 65:

        if(idx % 100 == 0):
            print('We have predicted the gene expression of', idx, 'genes')
        idx += 1

        batch.append(np.squeeze(bwh[idx]))

        if len(batch) == batch_size:
            for b in batch:
                x, y = b
                print("Shape of x", x.shape, "shape of y", y.shape)
                y_predicted = trained_model.predict(np.expand_dims(x, 0))
                print("Shape of y_predicted", y_predicted.shape)                    
                yTest.append(y)
                yPred.append(y_predicted)
            # Reset the batch
            batch = []


    # Now we output yTest and yPred
    print("Shape of yTest", np.asarray(yTest).shape, "shape of yPred", np.asarray(yPred).shape)