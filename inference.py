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


def modify_input(x, m, assay_type, batch_size, height, width, depth):
    # First split the data into epigenetics and sequence
    epigenetics = x[:, :height*2*width*depth]    
    seq = x[:, height*2*width*depth:]
    modified_seq = seq    

    # Now reshape the data to split into relevant axes
    epigenetics = epigenetics.reshape(batch_size, height, 2*width, depth)     
    modified_epigenetics = epigenetics
    assert(assay_type < depth)   

    '''
    # Permute the input features along the position dimension
    if(assay_type == -1): # permute the sequence
        seq = seq.reshape(batch_size, 4, 2*width*100)        
        seq = seq.transpose(0, 2, 1)
        [np.random.shuffle(x) for x in seq] # in-place permutation
        modified_seq = seq.transpose(0, 2, 1)
    else: # permute one of the 7 epigenetic assays 
        # First bring the position dimension next to the batch_size
        assay = epigenetics[:, :, :, assay_type].transpose(0, 2, 1)
        # Then consider each batch and apply same permutation across cell types
        [np.random.shuffle(x) for x in assay] # in-place permutation
        # Bring back the position to the third dimension 
        modified_assay = assay.transpose(0, 2, 1)
        modified_epigenetics = np.concatenate([epigenetics[:,:,:,:assay_type],
                               np.expand_dims(modified_assay, axis = 3),
                               epigenetics[:, :, :, assay_type+1:]], axis = 3)
    '''

    '''
    # Set the input feature to a constant value
    if(assay_type == -1): # set sequence to all Gs
        G = np.asarray([0, 0, 1, 0])
        modified_seq = np.repeat(np.expand_dims(np.repeat(np.expand_dims(G, 
                                                axis=0), 
                                                2*width*100, axis=0), axis=0),
                                                batch_size, axis=0)
        modified_seq = modified_seq.transpose(0, 2, 1) 
    else: # set p-value = 1e-103 => -log(p-value) = 103 => arcsinh(103-3) = 5.3
        modified_assay = np.full((batch_size, height, 2*width), 5.3)    
        modified_epigenetics = np.concatenate([epigenetics[:,:,:,:assay_type],
                               np.expand_dims(modified_assay, axis = 3),
                               epigenetics[:, :, :, assay_type+1:]], axis = 3)
    '''
    
    #'''
    # Create peaks that change by position for each assay
    modified_assay = np.full((batch_size, height, 2*width), -1.82)   
    modified_assay[:, :, m-10:m+10] = 5.3            
    modified_epigenetics = np.concatenate([epigenetics[:,:,:,:assay_type],
                           np.expand_dims(modified_assay, axis = 3),
                           epigenetics[:, :, :, assay_type+1:]], axis = 3)
    #'''

    '''
    # Multiply an assay_type by a supplied scalar
    modified_epigenetics = np.concatenate([epigenetics[:, :, :, :assay_type],
                np.expand_dims(m * epigenetics[:, :, :, assay_type], axis = 3),
                           epigenetics[:, :, :, assay_type+1:]], axis = 3)
    '''

    # print("modified epigenetics shape", modified_epigenetics.shape)
    # print("modified seq shape", modified_seq.shape)

    # Finally, reshape back 
    epigenetics = modified_epigenetics.reshape((batch_size, -1))
    seq = modified_seq.reshape(batch_size, -1)
    modified_x = np.hstack([epigenetics, seq])

    return modified_x


if __name__ == '__main__':

    batch_size = 16
    window_size = int(sys.argv[1])     
    bwh = BinnedHandlerSeqPredicting(window_size, batch_size, 
                                     drop_prob = 0, 
                                     CT_exchangeability=True)

    # We perform inference using a trained model
    model_number = 70

    # maximum_likelihood_loss(y_true, y_pred, num_output)    
    trained_model = load_model("model-"+str(model_number)+".hdf5",
                                custom_objects={'customLoss': customLoss})
    in_shape = trained_model.inputs[0].shape
    batch_size = int(in_shape[0])
    CT_exchangeability = True # True is what we used for EIC '19
    seg_len = None

    #  print("Input shape", in_shape, "Batch size", batch_size,
    #       "Window size", window_size)
    
    # We pick a random gene, and plot its predicted gene expression value,
    # as a function of the multiplier of different assays
    print('Beginning inference')

    cell_type = 10
    idx = 0
    for idx in range(100):
        x, y = bwh[idx]
        # x1 = np.expand_dims(x[0,:], axis=0)
        yTrue = y[0][cell_type]
        if ( (yTrue < 0.1)  ):
            continue

        yPred = []    


        for assay_type in range(0, 7): #(-1, 7) to include sequence also
            yy = []   
            for m_sign in [1]: #[-1, 1]

                # This is for multiplication scalars
                # for m in np.arange(-2.0, 2.1, 10):
                    # m = m_sign * pow(10, m)

                # This is for position of where the peaks will occur
                for m in np.arange(10, 190, 10):            
                    # print("Shape of x", x.shape, "shape of y", y.shape)        
                    y_predicted = trained_model.predict(modify_input(x, m,
                                                    assay_type,
                                                    batch_size, 
                                                    NUM_CELL_TYPES, 
                                                    window_size,
                                                    NUM_ASSAY_TYPES), 
                                                    batch_size=batch_size)
                    # print("Shape of y_predicted", y_predicted.shape)                                

                    yy.append(y_predicted[0][cell_type])
            yPred.append(yy)
        y_predicted_without_modifying = trained_model.predict(x)    
        print(yTrue, y_predicted_without_modifying[0][cell_type]) 
        print(np.squeeze(yPred))


