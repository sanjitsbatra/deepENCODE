from __future__ import print_function
from __future__ import division
import numpy as np
from os.path import join
from data_loader import BinnedHandlerSeqTraining
from data_loader import NUM_CELL_TYPES, NUM_ASSAY_TYPES
from models import customLoss, maximum_likelihood_loss
from numba import njit
from keras.models import load_model
import sys
import keras_genomics
from scipy.stats import spearmanr, pearsonr

import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()


def modify_input(x, m, assay_type, batch_size, height, width, depth, peak_value, peak_width):
    # First split the data into epigenetics and sequence
    epigenetics = x[:, :height*2*width*depth]    
    # print("Shape of x", x.shape, "Shape of epigenetics", epigenetics.shape)

    seq = x[:, height*2*width*depth:]
    modified_seq = seq    
    
    # Now reshape the data to split into relevant axes
    epigenetics = epigenetics.reshape(batch_size, height, 2*width, depth)     
    modified_epigenetics = epigenetics
    assert(assay_type < depth)   

    '''
    # Permute the input features along the position dimension
    if(assay_type == -1): # permute the sequence
        modified_seq = modified_seq.reshape(batch_size, 4, 2*width*100)        
        modified_seq = modified_seq.transpose(0, 2, 1)
        [np.random.shuffle(x) for x in modified_seq] # in-place permutation
        modified_seq = modified_seq.transpose(0, 2, 1)
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

    '''
    # Set the sequence to constant values in windows
    if(assay_type == -1): 
        G = np.asarray([0, 0, 1, 0])
        T = np.asarray([0, 0, 0, 1])
        modified_seq = modified_seq.reshape(batch_size, 4, 2*width*100)
        modified_seq = modified_seq.transpose(0, 2, 1)
        modified_seq[:, 100*(m-1):100*(m+1)] = np.repeat(np.expand_dims
                                               (np.repeat(np.expand_dims     
                                                (T, axis=0), 200, axis=0),
                                                axis=0), batch_size, axis=0)
        modified_seq = modified_seq.transpose(0, 2, 1)
    '''
    
    # '''
    # Create peaks that change by position for each assay
    # modified_assay = np.full((batch_size, height, 2*width), 0.01)   
    modified_assay = np.squeeze(epigenetics[:,:,:,assay_type])
    # print("Shape of modified assay", modified_assay.shape)
    if( (m-peak_width < 0) or (m+peak_width+1 >= 2*width) ):
        return x
    modified_assay[:, :, m-peak_width:m+peak_width+1] = peak_value            
    modified_epigenetics = np.concatenate([epigenetics[:,:,:,:assay_type],
                           np.expand_dims(modified_assay, axis = 3),
                           epigenetics[:, :, :, assay_type+1:]], axis = 3)
    # '''

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
    bwh = BinnedHandlerSeqTraining(window_size, batch_size, 
                                     drop_prob = 0, 
                                     CT_exchangeability=True)

    # maximum_likelihood_loss(y_true, y_pred, num_output)    
    model_number = int(sys.argv[2])
    model_name = "../Results/run_102_13_16_0/model-"+str(model_number)+".hdf5"
    trained_model = load_model(model_name,
                                custom_objects={'customLoss': customLoss})
    in_shape = trained_model.inputs[0].shape
    # print("Shape of inputs", in_shape)
    batch_size = int(in_shape[0])
    CT_exchangeability = True # True is what we used for EIC '19
    seg_len = None

    # print("Input shape", in_shape, "Batch size", batch_size,
    #        "Window size", window_size)
    
    # We pick a random gene, and plot its predicted gene expression value,
    # as a function of the multiplier of different assays
    #  print('Beginning inference')

    peak_value = float(sys.argv[3])    
    peak_width = int(sys.argv[4])
    cell_type = int(sys.argv[5])
    assay_type = int(sys.argv[6])
    # print("peak_value", str(peak_value))
    # print("peak_width", str(peak_width))

    output_prefix = str(model_number)+"."+str(peak_value)+"."+str(peak_width)+"."+str(cell_type)
    f_output = open("CXCR4_Report."+output_prefix+".txt", 'w')
    Alan_positions_CXCR4 = [-13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    idx = 0
    # print("Length of bwh is", len(bwh))
    for idx in range(len(bwh)):
        # print("idx is", str(idx))
        gene_names, x, y = bwh[idx]

        # CXCR4: ENSG00000121966 
        # TBX5: ENSG00000089225
        # IL1RN: ENSG00000136689
        special_gene = "ENSG00000121966"
        if(special_gene in gene_names):
            special_gene_index = gene_names.index(special_gene)
            # print("Found gene at", special_gene_index)         
        else:              
            # print(special_gene, "not found in", str(gene_names))
            continue
            # x1 = np.expand_dims(x[0,:], axis=0)

        yTrue = y[special_gene_index][cell_type]

        # if ( (yTrue < 1.00)  ):
        #     continue

        yPred = []    

        for assay in range(assay_type, assay_type+1): #(-1, 7) to include sequence also
            yy = []   
            for m_sign in [1]: #[-1, 1]

                # This is for multiplication scalars
                # for m in np.arange(-2.0, 2.1, 10):
                    # m = m_sign * pow(10, m)

                # This is for position of where the peaks will occur
                for m in np.arange(0, 30, 1):            
                    # print("Shape of x", x.shape, "shape of y", y.shape)        
                    y_predicted = trained_model.predict(modify_input(x, m,
                                                    assay,
                                                    batch_size, 
                                                    NUM_CELL_TYPES, 
                                                    window_size,
                                                    NUM_ASSAY_TYPES, 
                                                    peak_value,
                                                    peak_width), 
                                                    batch_size=batch_size)
                    # print("Shape of y_predicted", y_predicted.shape)                                

                    yy.append([m-window_size,
                            round(np.expm1(y_predicted[special_gene_index][cell_type]),3)])
            yPred.append(yy)
        y_predicted_without_modifying = trained_model.predict(x)    
        
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)
        print("yTrue\t"+str(yTrue)+"\nyPred\t"+str(round(np.expm1(y_predicted_without_modifying[special_gene_index][cell_type]),3)), file=f_output) 
        # print("yModified", np.squeeze(yPred))
        for e in np.squeeze(yPred):
            position = int(e[0])
            value = round(float(e[1]), 3)
            if(position in Alan_positions_CXCR4):
                print(str(position)+"\t"+str(value), file=f_output)        

        f_output.close()
        sys.exit(1)
    

