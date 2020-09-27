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


############## Define Keras backend helper functions ##################
# Non-negative mean for Permutation-Invariance
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


# Max for Permutation-Invariance
def keras_max(input_x):
    return K.max(input_x, axis=1, keepdims=True)


# Tiling across a dimension
def keras_tile(input_x, height):
    from keras import backend as K
    return K.tile(input_x, n=(1, height, 1, 1))


# Expand dimension of a tensor
def keras_expand_dims(input_x, axis):
    from keras import backend as K
    return K.expand_dims(input_x, axis=axis)


# Reshape a tensor to a given input shape
def keras_reshape(input_x, shape):
    from keras import backend as K
    return K.reshape(input_x, shape=shape)


# Swap axes of a tensor
def keras_transpose(input_x, axes):
    from keras import backend as K
    return K.permute_dimensions(input_x, axes)


# Squeeze dimensions of a tensor
def keras_squeeze(input_x):
    from keras import backend as K
    return K.squeeze(input_x, 0)


###################### END of Keras backend helper functions ##############


# This function performs convolutions on the sequence data only
def seq_module(OFFSET, batches, width, height, depth,
               seq_filters, num_seq_features, seg_len,
               batchnorm, inputs):

    # Print the dimensions of the input data
    print("height", height, "2*seg_len", 2*seg_len, "depth", depth) 

    # Subset the first few columns to obtain the epigenetic data of shape:
    # (BATCH_SIZE, NUM_CELL_TYPES*2*WINDOW_SIZE*NUM_ASSAY_TYPES)
    x = Lambda(lambda x: x[:, :height*2*seg_len*depth])(inputs)
    print('shape of x after taking first few columns= ', x.shape)

    # Reshape these columns to get the epigenetic data in the shape:
    # (BATCH_SIZE, NUM_CELL_TYPES, 2*WINDOW_SIZE, NUM_ASSAY_TYPES)
    x = Lambda(keras_reshape,
               arguments={'shape': (batches, height, 2*seg_len, depth)})(x)
    print('shape of x after reshape = ', x.shape)

    # Subset the rest of the columns to obtain the one-hot encoded sequence:
    # (BATCH_SIZE, 4 x 100 x 2*WINDOW_SIZE)
    seq = Lambda(lambda x: x[:, height*2*seg_len*depth:])(inputs)
    print('shape of seq = ', seq.shape)

    # Reshape this data to get 4 channels for the sequence in the shape:
    # (BATCH_SIZE, 4, 100 x 2*WINDOW_SIZE)  
    seq = Lambda(keras_reshape,
                 arguments={'shape': (batches, 4, 2*seg_len*100)})(seq)
    print('shape of seq after reshape = ', seq.shape)
    
    # Transpose dimensions to get:
    # (BATCH_SIZE, 100 x 2*WINDOW_SIZE, 4)
    seq = Lambda(keras_transpose,
                 arguments={'axes': (0, 2, 1)})(seq)
    print('shape of seq after transpose = ', seq.shape)

    # At this point, if we want to only look at a small window of the data
    # Let's say, [i*100bp, i*100bp + 10*100bp] where i could be in [-W, W]
    OFFSET_FLAG = 0
    if(OFFSET_FLAG==1):
        # OFFSET = 2
        LENGTH = 14
        
        slice_x = Lambda(lambda xx: xx[:, :, OFFSET:OFFSET + 2*LENGTH, :])
        slice_seq = Lambda(lambda ss: ss[:, 100 * OFFSET:100 * (OFFSET + 2 * LENGTH), :])
        x = slice_x(x)
        seq = slice_seq(seq)
        
        print("OFFSET-ed shapes are", x.shape, seq.shape)
        
        real_width = 100 * 2*LENGTH

    else:
        # Keep track of the width of the data
        real_width = 100 * 2*seg_len
        print('real width = '+str(real_width))

    # Perform convolutions on the sequence only, which has shape:
    # (BATCH_SIZE, 100 x 2*WINDOW_SIZE, 4)
    for filt in seq_filters:
        patch_width, patch_depth, dilate = filt

        seq = Conv1D(patch_depth,
                     patch_width,
                     dilation_rate=dilate,
                     padding='valid')(seq)

        if batchnorm:
            seq = BatchNormalization()(seq)
            # seq = RevCompConv1DBatchNorm()(seq)

        seq = Activation('elu')(seq)

        # seq = MaxPooling1D(pool_size=2)(seq) # Dilated convolutions instead

        # Keep track of width after each convolution
        real_width = real_width - (patch_width - 1) * dilate
        print('shape of seq after convolution ', filt, ' = ', seq.shape)

    # After performing the above convolutions
    # we want the sequence to be of shape:
    # (BATCH_SIZE, WIDTH, NUM_SEQ_FILTERS)
    # In order to get a prespecified WIDTH, we perform a final convolution
    WIDTH = 6
    seq = Conv1D(num_seq_features,
                 real_width-WIDTH+1,
                 strides=1,
                 padding='valid')(seq)
    if batchnorm:
        seq = BatchNormalization()(seq)
        # seq = RevCompConv1DBatchNorm()(seq)
    seq = Activation('elu')(seq)
    print('shape of seq after final 1D conv = ', seq.shape)

    # Now we add an extra dimension to the sequence to get:
    # (BATCH_SIZE, 1, WIDTH, NUM_SEQ_FILTERS)
    seq = Lambda(keras_expand_dims,
                 arguments={'axis': 1})(seq)
    print('shape of seq after expand dims = ', seq.shape)

    # We then tile the sequence to get:
    # (BATCH_SIZE, NUM_CELL_TYPES, WIDTH, NUM_SEQ_FILTERS) 
    seq = Lambda(keras_tile, arguments={'height': height})(seq)
    print('shape of seq after tiling = ', seq.shape)

    # We now return the processed sequence and unprocessed epigenetic data
    print('shape of seq at the end of processing it is = ', seq.shape)
    print('shape of unprocessed epigenetic data is = ', x.shape)
    return x, seq


# This function performs Permutation-Equivariant convolutions
def equivariant_layer(x, patch_width, patch_depth,
                       dilate, exch_func, padding, batchnorm):

    # TODO: Is this needed? 
    dilate = (1, dilate)

    # Permutation-Equivariance convolution
    x_equiv = Conv2D(patch_depth,
                   (1, patch_width),
                   dilation_rate=dilate,
                   padding=padding)(x)

    if batchnorm:
        x_equiv = BatchNormalization()(x_equiv)
    x_equiv = Activation('relu')(x_equiv) # relu is much better than linear
    print("Shape of Permutation-Equivariance after Conv2D", x_equiv.shape)

    return x_equiv


# This function performs Permutation-Invariant convolutions
def invariant_layer(x, patch_width, patch_depth,
                       dilate, exch_func, padding, batchnorm):

    # TODO: Is this needed? 
    dilate = (1, dilate)

    # Permutation-Invariant convolution
    x_inv = Conv2D(patch_depth,
                     (1, patch_width),
                     dilation_rate=dilate,
                     padding=padding)(x)
    if batchnorm:
        x_inv = BatchNormalization()(x_inv)
    x_inv = Activation('relu')(x_inv)

    print("Shape of Permutation-Invariance before Max", x_inv.shape)

    # Permutation-Invariant collapse
    if exch_func == 'max':
        x_inv = Lambda(keras_max)(x_inv)
    elif exch_func == 'nonneg_mean':
        x_inv = Lambda(keras_nonneg_mean)(x_inv)
    else:
        raise NotImplementedError(
            'exch_func must either be "max" or "nonneg_mean"')
    print("Shape of Permutation-Invariance after Max", x_inv.shape)

    # Permutation-Invariant tile
    x_inv = Lambda(keras_tile,
                     arguments={'height': int(x.shape[1])})(x_inv)
    print("Shape of Permutation-Invariance after tile", x_inv.shape)

    return x_inv


# This function performs convolutions on the epigenetic data 
# It then concatenates the processed sequence and epigenetic data
# and performs a final equivariant convolution to output NUM_CELL_TYPE scalars
def create_exchangeable_seq_cnn(OFFSET, batches, width, height, depth,
                                feature_filters=((11, 64, 1),
                                                 (11, 64, 1)),
                                seq_filters=((11, 128, 1),
                                             (11, 64, 1),
                                             (11, 32, 1)),
                                num_seq_features=32,
                                seg_len=None,
                                exch_func='max',
                                batchnorm=True,
                                density_network=False,
                                CT_exchangeability=True):
    if seg_len is None:
        seg_len = width

    input_shape = (batches, height*2*seg_len*depth + 100*4*2*seg_len)
    print('input_shape = ', input_shape)

    inputs = Input(batch_shape=input_shape)
    x, seq = seq_module(OFFSET, batches, width, height, depth, seq_filters,
                   num_seq_features, seg_len, batchnorm, inputs)
    WIDTH = 6

    #####################################################################################
    # Now perform equivariant convolutions
    x_equiv = x

    # Keep track of the width of the data
    real_width = int(x_equiv.shape[2])
    print('equivariant real width = '+str(real_width))
    
    # Epigenetic data currently has shape:
    # (BATCH_SIZE, NUM_CELL_TYPES, 2*WINDOW_SIZE, NUM_ASSAY_TYPES)
    print("Before Convs: Shape of x_equiv", x_equiv.shape, 
          "Shape of seq", seq.shape)
    for filter_params in feature_filters:
        patch_width, patch_depth, dilate = filter_params
        x_equiv = equivariant_layer(x_equiv, patch_width, patch_depth, 
                                    dilate, exch_func, 'valid', batchnorm)

        # Keep track of width after each convolution
        real_width = real_width - (patch_width - 1) * dilate
        print('shape of x_equiv after ', filter_params, ' equivariant = ', 
                x_equiv.shape)

    # Perform a final convolution on the epigenetic data to get WIDTH
    x_equiv = Conv2D(num_seq_features,
                 (1, real_width-WIDTH+1),
                 strides=1,
                 padding='valid')(x_equiv)
    if batchnorm:
        x_equiv = BatchNormalization()(x_equiv)
    x_equiv = Activation('relu')(x_equiv)
    print('shape of x_equiv after final 1D conv = ', x_equiv.shape)

    #####################################################################################
    # Now perform invariant convolutions
    x_inv = x

    # Keep track of the width of the data
    real_width = int(x_inv.shape[2])
    print('invariant real width = '+str(real_width))
    
    # Epigenetic data currently has shape:
    # (BATCH_SIZE, NUM_CELL_TYPES, 2*WINDOW_SIZE, NUM_ASSAY_TYPES)
    print("Before Convs: Shape of x_inv", x_inv.shape, 
          "Shape of seq", seq.shape)
    for filter_params in feature_filters:
        patch_width, patch_depth, dilate = filter_params
        x_inv = invariant_layer(x_inv, patch_width, patch_depth, dilate, 
                                exch_func, 'valid', batchnorm)

        # Keep track of width after each convolution
        real_width = real_width - (patch_width - 1) * dilate
        print('shape of x_inv after ', filter_params, 
              'invariant = ', x_inv.shape)

    # Perform a final convolution on the epigenetic data to get WIDTH
    x_inv = Conv2D(num_seq_features,
                 (1, real_width-WIDTH+1),
                 strides=1,
                 padding='valid')(x_inv)
    if batchnorm:
        x_inv = BatchNormalization()(x_inv)
    x_inv = Activation('relu')(x_inv)
    print('shape of x_inv after final 1D conv = ', x_inv.shape)

    #####################################################################################
    # After performing convolutions on the epigenetic data, its shape is:
    # (BATCH_SIZE, NUM_CELL_TYPES, WIDTH, NUM_FILTERS)
    # Further, the shape of the sequence data that we received was:
    # (BATCH_SIZE, NUM_CELL_TYPES, WIDTH, NUM_SEQ_FILTERS)
    print("Before combining, shape of x_equiv, x_inv and seq are", 
           x_equiv.shape, x_inv.shape, seq.shape)

    # Hence, we combine these modalities along the FILTERS dimension to get:
    # (BATCH_SIZE, NUM_CELL_TYPES, WIDTH, NUM_FILTERS + NUM_SEQ_FILTERS)
    x = Concatenate()([x_equiv, x_inv]) #, seq])
    print("After combining, the shape of x is", x.shape) 
  
    ######################################################################
    # Attempting to print the tensor x during training: FAILED 1 JULY 2020
    # x = K.print_tensor(x, message="Before final Conv2D")
    # tf.print(x)
    # print(K.get_value(x))
    # print(x.numpy()[0, 0, :, 0])
    # x_val = K.eval(x)
    # print(x_val)
    ######################################################################

    # In order to output exactly NUM_GENE_EXPRESSION_CELL_TYPES outputs
    # We have one filter at this step instead of NUM_ASSAY_TYPES filters
    if(CT_exchangeability):
        x = Conv2D(1, #NUM_ASSAY_TYPES,
                  (1, WIDTH),
                   padding='valid',
                   activation="linear")(x)
    else:
        x = Conv2D(1, #NUM_CELL_TYPES,
                  (1, WIDTH),
                   padding='valid',
                   activation="linear")(x)

    # This flatten IS essential: 28 JUNE 2020
    # Upon flatten-ing, we get:
    # (BATCH_SIZE, NUM_CELL_TYPES) instead of (BATCH_SIZE, NUM_CELL_TYPES,1,1)
    x = Flatten()(x)
    print('shape of x after final convolution = ', x.shape)

    model = Model(inputs, x)
    return model


# This function computes MSE ignoring entries that are -1000.0
def customLoss(yTrue, yPred):
    skip_indices = K.tf.where(K.tf.equal(yTrue, -1000.0), 
                              K.tf.zeros_like(yTrue),
                              K.tf.ones_like(yTrue))

    # For Regression
    loss = K.square(yTrue - yPred) * skip_indices
    return K.sum(loss, axis=-1) / K.sum(skip_indices, axis=-1)

    # For Classification (TODO: categorical crossentropy reports a bug)
    # loss = K.binary_crossentropy(yTrue, yPred) * skip_indices  
    # return K.sum(loss) / K.sum(skip_indices)


#########################END_OF_USEFUL_FUNCTIONS#######################


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
    input_shape = (batches, height*seg_len*depth + 100*4*seg_len)
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


