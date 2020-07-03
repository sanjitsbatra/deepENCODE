from __future__ import print_function
from keras.optimizers import Adam
from data_loader import BinnedHandlerSeqTraining
from data_loader import NUM_CELL_TYPES, NUM_ASSAY_TYPES
from models import create_exchangeable_seq_cnn
# from models import create_exchangeable_seq_resnet
from models import customLoss, maximum_likelihood_loss
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import sys
import os
from keras import backend as K
import numpy as np
# import keract # requires python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

NUM_CPU_THREADS = 5


def lr_scheduler(epoch):
    if epoch < 3:
        return 2e-3
    elif epoch < 45:
        return 1e-3
    else:
        return 5e-4


if __name__ == '__main__':

    # Setup model parameters
    seg_len = None

    batch_size = 64
    epochs = 100     
    steps_per_epoch = 100  

    run_name_prefix = sys.argv[1]
    window_size = int(sys.argv[2]) # => Length of window / 25 on each side
    num_filters = int(sys.argv[3])
    run_name = run_name_prefix+"_"+str(window_size)+"_"+str(num_filters)

    convolution_patch_width = 11
    model_type_flag = 'Regression' # 'Classification'
    drop_probability = 0.00
    CT_exchangeability = True # True is what we used for EIC19

    # Fix conv_length and num_filters to 4 or 8? and change dilation rate
    epigenetic_dilations =  [1, 1, 1, 2, 2, 2, 4, 4, 4, 8] 
    sequence_dilations = [1, 2, 4, 8, 16, 32, 64, 128, 228, 256, 256]

    feature_filters_input = [[convolution_patch_width, 
                              num_filters, epigenetic_dilations[i]]
                              for i in range(len(epigenetic_dilations))]

    seq_filters_input = [[convolution_patch_width, 
                              num_filters, sequence_dilations[i]]
                              for i in range(len(sequence_dilations))]

    # Keras fit generator through this class
    bwh = BinnedHandlerSeqTraining(window_size,
                                   batch_size,
                                   seg_len=seg_len,
                                   drop_prob=drop_probability,
                                   CT_exchangeability=CT_exchangeability)

    # Initialize model
    if(CT_exchangeability):
        model = create_exchangeable_seq_cnn(
        batch_size,
        window_size,
        NUM_CELL_TYPES,
        NUM_ASSAY_TYPES,
        feature_filters=feature_filters_input,
        seq_filters=seq_filters_input,
        num_seq_features=num_filters,
        seg_len=seg_len,
        exch_func='max',
        batchnorm=True,
        density_network=False,
        CT_exchangeability=CT_exchangeability)
    else:
        model = create_exchangeable_seq_cnn(
        batch_size,
        window_size,
        NUM_ASSAY_TYPES,
        NUM_CELL_TYPES,
        feature_filters=feature_filters_input,
        seq_filters=seq_filters_input,
        num_seq_features=num_filters,
        seg_len=seg_len,
        exch_func='max',
        batchnorm=True,
        density_network=False,
        CT_exchangeability=CT_exchangeability)
        
    # Compile model with custom MSE loss that skips missing entries
    opt = Adam(lr=1e-3)
    model.compile(loss=customLoss, optimizer=opt)

    # Display model
    print(model.summary())
    # print(model.layers)

    # Setup paths and callbacks
    if not os.path.exists(run_name):
        os.mkdir(run_name)
    print('Created ', str(run_name), ' directory')
    filepath = run_name+'/model-{epoch:02d}.hdf5'
    checkpoint = ModelCheckpoint(filepath)
    lr_schedule = LearningRateScheduler(lr_scheduler)
    callbacks_list = [lr_schedule, checkpoint]

    # Train model
    TRAIN_FLAG = 1
    if(TRAIN_FLAG == 1):
        # We then train with this batch of data
        model.fit_generator(generator=bwh,
                            steps_per_epoch=steps_per_epoch,
                            epochs=epochs,
                            callbacks=callbacks_list,
                            use_multiprocessing=False)
                            # workers=NUM_CPU_THREADS,
                            # max_queue_size=100)
  
    # Special module created to debug layer-wise outputs
    DEBUG_LAST_LAYER = 0
    if(DEBUG_LAST_LAYER == 1):

   
        # This has demonstrated (29 June 2020) that:
        # 1) With x_equiv channels, the untrained model outputs different 
        # values for different cell types
        # 2) and that with x_inv, the untrained model outputs identical 
        # values for different cell types

        # print("training round")
        # keras_function = K.function([model.input], [model.layers[-2].output])
        # print(keras_function([np.full((12,40,7), 1.0), 1]))

        inp = model.input
        outputs = [layer.output for layer in [model.get_layer('lambda_2')]]
        functor = K.function([inp, K.learning_phase()], outputs ) 

        example_input = np.asarray(
                        [np.full((2*window_size, NUM_ASSAY_TYPES), i) 
                         for i in range(6,6+NUM_CELL_TYPES)])
        example_input = example_input.reshape(-1)
        print("example_input shape", example_input.shape)

        example_seq_input = np.zeros((25*2*window_size, 4))
        example_seq_input = example_seq_input.reshape(-1)
        print("example_seq_input shape", example_seq_input.shape)

        example_input = np.hstack([example_input, example_seq_input])
        print("concatenated shape", example_input.shape)
   
        example_input = np.repeat(example_input[np.newaxis,...], 
                                  batch_size, axis=0)
        print("final shape", example_input.shape)

        print("Conv2D_1 layer", model.get_layer('conv2d_1'))

        output_1 = np.asarray(functor([example_input, 1.]))
        print(output_1.shape)
        print(output_1[0, 0, 0, :, 0])

        print(model.predict(example_input))



        # activations = keract.get_activations(model, 
        #                                     example_input,  
        #                                    layer_name='conv2d_11',
        #                                    auto_compile=True)
        # [print(k, '->', v.shape, '- Numpy array') for (k, v) in activations.items()]


    print('Training has completed! Exiting')
    os._exit(1)
