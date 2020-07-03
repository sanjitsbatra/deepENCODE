from __future__ import print_function
from __future__ import division
import numpy as np
from os.path import join
from data_loader import BinnedHandlerSeqPredicting
from data_loader import NUM_CELL_TYPES, NUM_ASSAY_TYPES
from models import customLoss, maximum_likelihood_loss
from numba import njit
from keras.models import load_model
import sys
import keras_genomics
from scipy.stats import spearmanr, pearsonr


if __name__ == '__main__':

    # maximum_likelihood_loss(y_true, y_pred, num_output)    
    trained_model = load_model(sys.argv[1],
                                custom_objects={'customLoss': customLoss})
    # 'RevCompConv1D': keras_genomics.layers.RevCompConv1D,
    # 'RevCompConv1DBatchNorm': 
    #       keras_genomics.layers.normalization.RevCompConv1DBatchNorm})

    in_shape = trained_model.inputs[0].shape
    batch_size = int(in_shape[0])
    window_size = int(sys.argv[2]) 
    CT_exchangeability = True # True is what we used for EIC '19
    seg_len = None

    print("Input shape", in_shape, "Batch size", batch_size,
          "Window size", window_size)

    # Don't dropout any epigenetic tracks at prediction time
    bwh = BinnedHandlerSeqPredicting(window_size, batch_size, 
                                     drop_prob = 0, 
                                     CT_exchangeability=CT_exchangeability)
    print('Beginning prediction')

    yTrue = []
    yPred = []

    len_bwh = len(bwh)
    print("Number of genes = ", len_bwh) 

    idx = 0
    while idx < 2*int(len_bwh / batch_size):

        if(idx % 100 == 0):
            print('We have predicted the gene expression of',
                  idx * batch_size, 'genes. Progress: ',
                  ((idx * batch_size) * 100.0) / 2*len_bwh, '%')

        idx += 1

        x, y = bwh[idx]
        # print(x[0, 1:100])

        print("Shape of x", x.shape, "shape of y", y.shape)
        yTrue.append(y)
        
        y_predicted = trained_model.predict(x, batch_size=batch_size)
        yPred.append(y_predicted)
        print("Shape of y_predicted", y_predicted.shape)                    

    # Now we output yTrue and yPred
    yTrue = np.vstack(yTrue)
    yPred = np.vstack(yPred)
    np.savetxt("yTrue.txt", yTrue, fmt='%1.4f')
    np.savetxt("yPred.txt", yPred, fmt='%1.4f')
    print("Shape of yTrue", yTrue.shape, "shape of yPred", yPred.shape)

    print("yTrue[10,:]", yTrue[10,:])
    print("yPred[10,:]", yPred[10,:])

    print("yTrue[20,:]", yTrue[20,:])
    print("yPred[20,:]", yPred[20,:])

    # Compute some statistics on yTrue and yPred
    assert(yTrue.shape[0] == yPred.shape[0])
    for i in range(yTrue.shape[1]):
        for j in range(yPred.shape[1]):
            cor = spearmanr(yTrue[:,i], yPred[:,j])[0]
            print(np.mean(yTrue[:,i]), np.mean(yPred[:,j]), i, j, cor)
            print(np.std(yTrue[:,i]), np.std(yPred[:,j]), i, j, cor)

    print("Finished predicting!")
