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
    batch_size = 8
    window_size = 90 # => Length of window / 25 on each side line 171 of data_loader
    seg_len = None
    steps_per_epoch = 100  
    epochs = 100     

    num_conv = int(sys.argv[2])
    num_seq_conv = int(sys.argv[3])
    num_filters = int(sys.argv[4])
    model_type_flag = 'Regression' # 'Classification'
    drop_probability = float(sys.argv[5])

    num_seq_filters = int(sys.argv[6])
    CT_exchangeability = int(sys.argv[7]) > 0 # True is what we used for EIC19
 
    # This keeps obtaining a random batch of data
    # bwh = BinnedHandlerTraining(window_size, batch_size)
    bwh = BinnedHandlerSeqTraining(window_size,
                                   batch_size,
                                   seg_len=seg_len,
                                   drop_prob=drop_probability,
                                   CT_exchangeability=CT_exchangeability)

    
    # TODO: what to print since len function isn't implemented
    # print(len(bwh))

    # Check to make sure this wasn't accidentally True
    density_network = False
    if model_type_flag == 'Regression':
        density_network = False
    elif model_type_flag == 'Classification':
        density_network = False
    else:
        assert model_type_flag == 'density'

    # Fix conv_length and num_filters to 4 or 8? and change dilation rate
    feature_filters_input = [[7, num_filters, 1]]*num_conv
    seq_filters_input = [[7, num_filters, 1]]*num_seq_conv

    if(CT_exchangeability):
        model = create_exchangeable_seq_cnn(
        batch_size,
        window_size,
        NUM_CELL_TYPES,
        NUM_ASSAY_TYPES,
        feature_filters=feature_filters_input,
        seq_filters=seq_filters_input,
        num_seq_features=num_seq_filters,
        seg_len=seg_len,
        exch_func='max',
        batchnorm=True,
        density_network=density_network,
        CT_exchangeability=CT_exchangeability)
    else:
        model = create_exchangeable_seq_cnn(
        batch_size,
        window_size,
        NUM_ASSAY_TYPES,
        NUM_CELL_TYPES,
        feature_filters=feature_filters_input,
        seq_filters=seq_filters_input,
        num_seq_features=num_seq_filters,
        seg_len=seg_len,
        exch_func='max',
        batchnorm=True,
        density_network=density_network,
        CT_exchangeability=CT_exchangeability)
        
    opt = Adam(lr=1e-3)
    if density_network:
        def loss(a, b):
            num_output = ((seg_len - window_size + 1)
                          * NUM_CELL_TYPES
                          * NUM_ASSAY_TYPES)
            return maximum_likelihood_loss(a, b, num_output)

        model.compile(loss=loss, optimizer=opt)
    else:
        model.compile(loss=customLoss, optimizer=opt)

    print(model.summary())

    # print(model.layers)

    run_name = sys.argv[1]
    if not os.path.exists(run_name):
        os.mkdir(run_name)
    print('Created ', str(run_name), ' directory')
    filepath = run_name+'/model-{epoch:02d}.hdf5'
    checkpoint = ModelCheckpoint(filepath)
    lr_schedule = LearningRateScheduler(lr_scheduler)
    callbacks_list = [lr_schedule, checkpoint]

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

        example_input = np.asarray(
                        [np.full((40, 7), i) for i in range(6+1,6+13)])
        example_input = example_input.reshape(-1)
        print("example_input shape", example_input.shape)

        example_seq_input = np.zeros((1000, 4))
        example_seq_input = example_seq_input.reshape(-1)
        print("example_seq_input shape", example_seq_input.shape)

        example_input = np.hstack([example_input, example_seq_input])
        print("concatenated shape", example_input.shape)
   
        example_input = np.expand_dims(example_input, axis = 0)
        print("final shape", example_input.shape)

        print("Conv2D_11 layer", model.get_layer('conv2d_11'))

        print(model.predict(example_input))


        # activations = keract.get_activations(model, 
        #                                     example_input,  
        #                                    layer_name='conv2d_11',
        #                                    auto_compile=True)
        # [print(k, '->', v.shape, '- Numpy array') for (k, v) in activations.items()]

        sys.exit(-4)


    # We then train with this batch of data
    model.fit_generator(generator=bwh,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        callbacks=callbacks_list,
                        use_multiprocessing=False)
                        # workers=NUM_CPU_THREADS,
                        # max_queue_size=100)
  


    print('Training has completed! Exiting')
    os._exit(1)
