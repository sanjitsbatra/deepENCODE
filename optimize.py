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

    loss = -1*K.square(yDesired - yPred[0][cell_type])
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
    cell_type = 4
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
    yDesired = yTrue + 3
    yPred = 100

    for iteration in range(100):
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

    x_0 = x_0.reshape(1, NUM_CELL_TYPES, 2*window_size, NUM_ASSAY_TYPES)
    xx = xx.reshape(1, NUM_CELL_TYPES, 2*window_size, NUM_ASSAY_TYPES)
    for at in range(NUM_ASSAY_TYPES):
        print("TRUE", at+1, x_0[:, cell_type, :, at])
        print("Optimized", at+1, xx[:, cell_type, :, at])   
    
