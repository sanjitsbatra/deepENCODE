from __future__ import print_function
from __future__ import division

from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Flatten
from keras.layers import Input, Concatenate 
from keras.layers import MaxPooling1D, Lambda, Add
from keras.layers.convolutional import Conv1D, Conv2D
from keras.models import Model
# from keras.losses import logcosh
import tensorflow as tf
from keras import backend as K
from data_loader import NUM_CELL_TYPES, NUM_ASSAY_TYPES
# from keras_genomics.layers.normalization import RevCompConv1DBatchNorm
# from keras_genomics.layers import RevCompConv1D
import numpy

def keras_nonneg_mean(input_x):
    num_nonneg = K.tf.math.reduce_sum(
        K.tf.cast(K.tf.math.greater_equal(input_x, 0.), K.tf.float32),
        axis=1,
        keepdims=True)
    sums = K.tf.math.reduce_sum(
        K.tf.nn.relu(input_x),
        axis=1,
        keepdims=True)

    # Need to avoid division by zero.
    # If denominator is zero, numerator is also zero by construction.
    # Adding a small positive constant to the denominator will not
    # affect the mean when the denominator is >= 1, but will prevent
    # division by 0, causing 0/0 = 0
    return sums / (num_nonneg + 1e-16)


def keras_max(input_x):
    return K.max(input_x, axis=1, keepdims=True)


def keras_tile(input_x, height):
    from keras import backend as K
    return K.tile(input_x, n=(1, height, 1, 1))


def keras_expand_dims(input_x, axis):
    from keras import backend as K
    return K.expand_dims(input_x, axis=axis)


def keras_reshape(input_x, shape):
    from keras import backend as K
    return K.reshape(input_x, shape=shape)


def keras_transpose(input_x, axes):
    from keras import backend as K
    return K.permute_dimensions(input_x, axes)


def keras_squeeze(input_x):
    from keras import backend as K
    return K.squeeze(input_x, 0)


def exchangeable_layer(x, patch_width, patch_depth,
                       dilate, exch_func, padding, batchnorm):
    dilate = (1, dilate)

    # Permutation-Invariance
    x_inv = Conv2D(patch_depth,
                     (1, patch_width),
                     dilation_rate=dilate,
                     padding=padding)(x)
    if batchnorm:
        x_inv = BatchNormalization()(x_inv)
    x_inv = Activation('relu')(x_inv)

    print("Shape of Permutation-Invariance before Max", x_inv.shape)
    if exch_func == 'max':
        x_inv = Lambda(keras_max)(x_inv)
    elif exch_func == 'nonneg_mean':
        x_inv = Lambda(keras_nonneg_mean)(x_inv)
    else:
        raise NotImplementedError(
            'exch_func must either be "max" or "nonneg_mean"')
    print("Shape of Permutation-Invariance after Max", x_inv.shape)

    x_inv = Lambda(keras_tile,
                     arguments={'height': int(x.shape[1])})(x_inv)
    print("Shape of Permutation-Invariance after tile", x_inv.shape)

    # Permutation-Equivariance
    x_equiv = Conv2D(patch_depth,
                   (1, patch_width),
                   dilation_rate=dilate,
                   padding=padding)(x)

    if batchnorm:
        x_equiv = BatchNormalization()(x_equiv)
    x_equiv = Activation('relu')(x_equiv) # relu is much better than linear
    print("Shape of Permutation-Equivariance after Conv2D", x_equiv.shape)

    x = Concatenate()([x_inv, x_equiv])

    return x


def seq_module(batches, width, height, depth,
               seq_filters, num_seq_features, seg_len,
               batchnorm, inputs):

    print("height", height, "2*seg_len", 2*seg_len, "depth", depth) 

    x = Lambda(lambda x: x[:, :height*2*seg_len*depth])(inputs)
    print('shape of x after taking first few columns= ', x.shape)

    x = Lambda(keras_reshape,
               arguments={'shape': (batches, height, 2*seg_len, depth)})(x)
    print('shape of x after reshape = ', x.shape)

    seq = Lambda(lambda x: x[:, height*2*seg_len*depth:])(inputs)
    print('shape of seq = ', seq.shape)

    seq = Lambda(keras_reshape,
                 arguments={'shape': (batches, 4, 2*seg_len*25)})(seq)
    print('shape of seq after reshape = ', seq.shape)

    seq = Lambda(keras_transpose,
                 arguments={'axes': (0, 2, 1)})(seq)
    print('shape of seq after transpose = ', seq.shape)

    for filt in seq_filters:
        patch_width, patch_depth, dilate = filt
        seq = Conv1D(patch_depth,
        # seq = RevCompConv1D(patch_depth,
                            patch_width,
                            dilation_rate=dilate,
                            padding='same')(seq)
        if batchnorm:
            seq = BatchNormalization()(seq)
            # seq = RevCompConv1DBatchNorm()(seq)
        seq = Activation('relu')(seq)
        seq = MaxPooling1D(pool_size=2)(seq)
        print('shape of seq after convolution ', filt, ' = ', seq.shape)

    seq = Conv1D(num_seq_features,
    # seq = RevCompConv1D(num_seq_features,
                        22,
                        strides=1,
                        padding='valid')(seq)
    if batchnorm:
        seq = BatchNormalization()(seq)
        # seq = RevCompConv1DBatchNorm()(seq)
    seq = Activation('relu')(seq)
    print('shape of seq after final 1D conv = ', seq.shape)

    seq = Lambda(keras_expand_dims,
                 arguments={'axis': 1})(seq)
    print('shape of seq after expand dims = ', seq.shape)

    seq = Lambda(keras_tile, arguments={'height': height})(seq)
    print('shape of seq after tiling = ', seq.shape)

    #####################
    # x = x #Concatenate()([x, seq])
    print('shape of seq at the end of processing it is = ', seq.shape)
    return x, seq

def create_exchangeable_seq_cnn(batches, width, height, depth,
                                feature_filters=((11, 64, 1),
                                                 (11, 64, 1)),
                                seq_filters=((11, 128, 1),
                                             (11, 64, 1),
                                             (11, 32, 1)),
                                num_seq_features=32,
                                seg_len=None,
                                exch_func='max',
                                batchnorm=False,
                                density_network=False,
                                CT_exchangeability=True):
    if seg_len is None:
        seg_len = width

    input_shape = (batches, height*2*seg_len*depth + 25*4*2*seg_len)
    print('input_shape = ', input_shape)

    inputs = Input(batch_shape=input_shape)
    x, seq = seq_module(batches, width, height, depth, seq_filters,
                   num_seq_features, seg_len, batchnorm, inputs)

    real_width = 2*seg_len
    print('real width = '+str(real_width))

    print("Before Convs: Shape of x", x.shape, "Shape of seq", seq.shape)

    for filter_params in feature_filters:
        patch_width, patch_depth, dilate = filter_params
        x = exchangeable_layer(x, patch_width, patch_depth, dilate, exch_func,
                               'valid', batchnorm)

        # Keep track of real width after each convolution
        real_width = real_width - (patch_width - 1) * dilate
        
        print('shape of x after ', filter_params, ' exchangeable = ', x.shape)

    # In order to output exactly NUM_GENE_EXPRESSION_CELL_TYPES outputs
    # We have one filter at this step instead of NUM_ASSAY_TYPES filters
    

    print("Before combining, shape of x and seq are", x.shape, seq.shape)
    x = x # seq #Concatenate()([x, seq])
    print("After combinging, the shape of x is", x.shape)
    
    print("Right before final Conv2D per row, real_width", real_width)
    print("And the shape of x is", x.shape)

    # x = K.print_tensor(x, message="Before final Conv2D")
    # tf.print(x)
    # print(K.get_value(x))
    # print(x.numpy()[0, 0, :, 0])
    # x_val = K.eval(x)
    # print(x_val)

    ############### Is this correct?
    # For Regression:
    if(CT_exchangeability):
        x_mu = Conv2D(1, #NUM_ASSAY_TYPES,
              (1, real_width),
              padding='valid',
              activation="linear")(x)
    else:
        x_mu = Conv2D(1, #NUM_CELL_TYPES,
              (1, real_width),
                      padding='valid',
              activation="linear")(x)

    	

    # For Classification
    # if(CT_exchangeability):
    #     x_mu = Conv2D(NUM_ASSAY_TYPES,
    #                   (1, real_width),
    #                   padding='valid',
    #                   activation="sigmoid")(x)
    # else:
    #     x_mu = Conv2D(NUM_CELL_TYPES,
    #                   (1, real_width),
    #                   padding='valid',
    #                   activation="sigmoid")(x)

    if density_network:
        if(CT_exchangeability):
            x_log_precision = Conv2D(NUM_ASSAY_TYPES,
                                     (1, real_width),
                                     padding='valid')(x)
        else:
            x_log_precision = Conv2D(NUM_CELL_TYPES,
                                     (1, real_width),
                                     padding='valid')(x)
        x_log_precision = Flatten()(x_log_precision)
        x = Concatenate()([x_mu, x_log_precision])
    else:
        # (None, 51, 1, 100)
        # x_mu = Conv2D(1, (1,1), padding="same")(x_mu)

        ####################### Can we skip the flatten here?
        # Flatten IS essential: 30 June 2020
        x = Flatten()(x_mu) #K.squeeze(K.squeeze(x_mu, axis=3), axis=2) # Flatten()(x_mu)
        # pass
    
    # print(x)

    print('shape of x after final convolution = ', x.shape)
    model = Model(inputs, x)
    return model


def customLoss(yTrue, yPred):
    skip_indices = K.tf.where(K.tf.equal(yTrue, -1000.0), K.tf.zeros_like(yTrue),
                              K.tf.ones_like(yTrue))

    # For Regression
    loss = K.square(yTrue - yPred) * skip_indices
    return K.sum(loss, axis=-1) / K.sum(skip_indices, axis=-1)

    # For Classification (TODO: categorical crossentropy reports a bug)
    # loss = K.binary_crossentropy(yTrue, yPred) * skip_indices  
    # return K.sum(loss) / K.sum(skip_indices)


def maximum_likelihood_loss(y_true, y_pred, num_output):
    mu = y_pred[:, :num_output]
    log_precision = y_pred[:, num_output:] / 1e3
    skip_indices = K.tf.where(K.tf.equal(y_true, -1000.0),
                              K.tf.zeros_like(y_true),
                              K.tf.ones_like(y_true))
    like = skip_indices * (K.square(y_true - mu) * K.exp(log_precision) -
                           log_precision)
    return K.mean(like)


def create_exchangeable_seq_resnet(batches, width, height, depth,
                                   feature_filters=((5, 1),) * 10,
                                   seq_filters=((10, 8, 1),
                                                (10, 16, 1),
                                                (10, 32, 1),
                                                (10, 64, 1)),
                                   num_seq_features=65,
                                   seg_len=None,
                                   exch_func='nonneg_mean',
                                   batchnorm=False,
                                   CT_exchangeability=True):

    if seg_len is None:
        seg_len = width
    input_shape = (batches, height*seg_len*depth + 25*4*seg_len)
    print('input_shape = ', input_shape)
    inputs = Input(batch_shape=input_shape)
    x_depth = num_seq_features + depth
    x = seq_module(batches, width, height, depth, seq_filters,
                   num_seq_features, seg_len, batchnorm, inputs)
    for filter_params in feature_filters:
        patch_width, dilate = filter_params
        y = exchangeable_layer(x, patch_width, x_depth,
                               dilate, exch_func, 'same', batchnorm)
        y = exchangeable_layer(y, patch_width, x_depth,
                               dilate, exch_func, 'same', batchnorm)
        # The first x_depth channels are equivariant
        # the next x_depth channels are invariant
        # because of the res_net architecture, we don't
        # want which channels are invariant and equivariant
        # to matter, so we'll just collapse them via summation
        y = Lambda(lambda y: y[:, :, :, :x_depth] + y[:, :, :, x_depth:])(y)
        x = Add()([x, y])
    if(CT_exchangeability):
        x = Conv2D(NUM_ASSAY_TYPES,
                   (1, width),
                   padding='valid')(x)
    else:
        x = Conv2D(NUM_CELL_TYPES,
                   (1, width),
                   padding='valid')(x)
    print('shape of x after final convolution = ', x.shape)

    x = Flatten()(x)
    model = Model(inputs, x)
    return model


def create_exchangeable_cnn(batches, width, height, depth,
                            filters=((5, 2*NUM_ASSAY_TYPES, 1),
                                     (5, 3*NUM_ASSAY_TYPES, 1)),
                            seg_len=None,
                            exch_func='max',
                            CT_exchangeability=True):
    if seg_len is None:
        seg_len = width

    # We assume that we have already reshaped x so that assays are channels
    # height is n, width is l, depth is m

    input_shape = (batches, height, seg_len, depth)

    # define the model input
    inputs = Input(batch_shape=input_shape)

    # Perform multiple convolutional layers
    real_width = width
    for i, filter_params in enumerate(filters):
        patch_width, patch_depth, dilate = filter_params
        if i == 0:
            x = inputs
        x = exchangeable_layer(x, patch_width, patch_depth, dilate, exch_func,
                               'valid')
        real_width = real_width - (patch_width - 1) * dilate

    # Now we add a dense layer
    if(CT_exchangeability):
        x = Conv2D(NUM_ASSAY_TYPES,
                   (1, real_width),
                   padding='valid')(x)
    else:
        x = Conv2D(NUM_CELL_TYPES,
                   (1, real_width),
                   padding='valid')(x)
    # We want B * n * m
    x = Flatten()(x)

    # construct the CNN
    model = Model(inputs, x)

    # return the CNN
    return model
