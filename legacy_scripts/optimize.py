from __future__ import print_function
from __future__ import division
import sys, os
import numpy as np

import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from data_loader import BinnedHandlerSeqTraining
from data_loader import NUM_CELL_TYPES, NUM_ASSAY_TYPES
from models import customLoss


def compute_loss(trained_model, x, x_0, yPred, yDesired, cell_type, lambdaa):
    xx = tf.concat([x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x], 0)
    yPred = trained_model(xx, training=False)

    loss = (-1*K.square(yDesired - yPred[0][0]) + 
            -1*K.square(yDesired - yPred[0][1]) +
            -1*K.square(yDesired - yPred[0][2]) +
            -1*K.square(yDesired - yPred[0][3]) +
            -1*K.square(yDesired - yPred[0][4]) +
            -1*K.square(yDesired - yPred[0][5]) +
            -1*K.square(yDesired - yPred[0][6]) +
            -1*K.square(yDesired - yPred[0][7]) +
            -1*K.square(yDesired - yPred[0][8]) +
            -1*K.square(yDesired - yPred[0][9]) +
            -1*K.square(yDesired - yPred[0][10]) +
            -1*K.square(yDesired - yPred[0][11]) )
            
    print("Computed loss")
    tf.print(loss)
    return loss #tf.reduce_mean(x)


@tf.function
def gradient_ascent_step(trained_model, x, x_0, 
                        yPred, yDesired, cell_type, lambdaa):
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = compute_loss(trained_model, x, x_0, 
                            yPred, yDesired, cell_type, lambdaa)
        
        # Compute gradients
        grads = tape.gradient(loss, x)
        print("Computed gradients shape", grads.shape)
        # tf.print(grads)

        # Normalize gradients
        grads = tf.math.l2_normalize(grads)

        x += 1e-1 * grads # learning rate
    return loss, x 


if __name__ == '__main__':

    lambdaa = float(sys.argv[2])
    cell_type = 10
    model_number = 35
    trained_model = load_model("model-"+str(model_number)+".hdf5",
                                custom_objects={'customLoss': customLoss})

    in_shape = trained_model.inputs[0].shape
    batch_size = int(in_shape[0])    
    window_size = int(sys.argv[1])     
    bwh = BinnedHandlerSeqTraining(window_size, batch_size, 
                                     drop_prob = 0, 
                                     CT_exchangeability=True)

    X, Y = bwh[42]
    x_0 = np.expand_dims(X[0], axis=0)
    yTrue = Y[0][cell_type]
    print(x_0.shape, yTrue.shape)

    x = np.copy(x_0)
    x = tf.convert_to_tensor(x)
    yDesired = yTrue + 5
    yPred = 100

    for iteration in range(300):
        losss, x = gradient_ascent_step(trained_model, x, x_0, yPred, yDesired,
                                        cell_type, lambdaa)

    # Now we plot how the features of this cell type changed
    print(x_0.shape)
    print(x.shape)    
    xx = x.numpy() #tf.make_ndarray(x)
    print(xx.shape)    

    predicted_x_0 = trained_model.predict(np.tile(x_0, [16, 1]))[0][cell_type]
    predicted_xx =  trained_model.predict(np.tile(xx, [16, 1]))[0][cell_type]

    print("yTrue", yTrue, "yTrueP", predicted_x_0)
    print("yDesired", yDesired, "yOptimized", predicted_xx)

    # os._exit(0) 
    # sys.exit(0)
    
    f = open("output."+str(predicted_x_0)+"_"+str(predicted_xx)+".txt", 'w')

    x_0 = x_0.reshape(1, NUM_CELL_TYPES, 2*window_size, NUM_ASSAY_TYPES)
    xx = xx.reshape(1, NUM_CELL_TYPES, 2*window_size, NUM_ASSAY_TYPES)
    for at in range(NUM_ASSAY_TYPES):
        for pos in range(2*window_size):
            print(at+1, pos+1, (xx[:, cell_type, pos, at][0] - 
                                    x_0[:, cell_type, pos, at][0]) , file=f)

    # for at in range(NUM_ASSAY_TYPES):
    #     for pos in range(2*window_size):
    #      print("OPTIMIZED",at+1,pos+1,xx[:, cell_type, pos, at][0], file=f)

    f.close()
    
