import sys
import os
from chrmt_generator import EpigenomeGenerator, TranscriptomeGenerator
from chrmt_generator import ASSAY_TYPES, MASK_VALUE, EPS
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Activation, Dropout
from tensorflow.keras.layers import Conv1D, Input, add
# from tensorflow.keras.layers import Cropping1D
from tensorflow.keras.layers import Flatten, Dense
# from keras.losses import logcosh
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras import backend as K
from tqdm.keras import TqdmCallback
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


DROPOUT_PROB = 0.75


def lr_scheduler(epoch):

    if epoch < 3:
        return 2e-3
    elif epoch < 90:
        return 1e-3
    elif epoch < 200:
        return 5e-4
    else:
        return 1e-4


# Compute a loss that incorporates both mean and variance
def maximum_likelihood_loss(yTrue, yPred):

    DEBUG = False
    if(DEBUG):
        yTrue = K.print_tensor(yTrue, message='yTrue = ')

    yTrue_flattened = K.flatten(yTrue)

    if(DEBUG):
        yTrue_flattened = K.print_tensor(yTrue_flattened,
                                         message='yTrue_fl = ')
        yPred = K.print_tensor(yPred, message='yPred = ')

    yPred_mean = yPred[:, 0]

    if(DEBUG):
        yPred_mean = K.print_tensor(yPred_mean, message='pm = ')

    yPred_log_precision = yPred[:, 1] + EPS

    if(DEBUG):
        yPred_log_precision = K.print_tensor(yPred_log_precision,
                                             message='ylp = ')

    squared_loss = K.square(yTrue_flattened - yPred_mean)

    if(DEBUG):
        squared_loss = K.print_tensor(squared_loss, message='squared loss = ')

    loss = K.mean(squared_loss * K.exp(yPred_log_precision), axis=-1)
    loss = loss - 1.0 * K.mean(yPred_log_precision, axis=-1)

    if(DEBUG):
        loss = K.print_tensor(loss, message='loss = ')

    return loss


# Compute an MSE loss for LM only at those positions that are NOT MASK_VALUE
def custom_loss(yTrue, yPred):

    masked_indices = K.tf.where(K.tf.equal(yTrue, MASK_VALUE),
                                K.tf.zeros_like(yTrue),
                                K.tf.ones_like(yTrue))

    '''
    # MSE loss
    loss = K.square((yTrue-yPred) * masked_indices)
    sum_numerator = K.sum(K.sum(loss, axis=-1), axis=-1, keepdims=True)
    sum_denominator = K.sum(K.sum(masked_indices, axis=-1), axis=-1,
                            keepdims=True)
    mse_loss = sum_numerator / (sum_denominator + EPS)
    '''

    # logcosh loss
    def _logcosh(x):
        return x + K.softplus(-2. * x) - K.log(2.)
    logcosh_loss = K.mean(K.mean(_logcosh((yTrue - yPred) * masked_indices),
                          axis=-1, keepdims=False), axis=-1, keepdims=True)

    return logcosh_loss


def BAC(x_input, conv_kernel_size, num_filters):

    # x_output = BatchNormalization()(x_input)
    # x_output = Activation('relu')(x_output)
    x_output = Conv1D(kernel_size=conv_kernel_size,
                      filters=num_filters,
                      padding='same')(x_input)
    x_output = BatchNormalization()(x_output)
    x_output = Activation('relu')(x_output)
    x_output = BatchNormalization()(x_output)
    return x_output


def residual_block(x, conv_kernel_size, num_filters):

    x_1 = BAC(x, conv_kernel_size, num_filters)
    x_2 = BAC(x_1, conv_kernel_size, num_filters)
    output = Dropout(DROPOUT_PROB)(add([x, x_2]))

    return output


def create_epigenome_cnn(number_of_assays,
                         window_size,
                         num_filters,
                         conv_kernel_size,
                         num_convolutions,
                         padding):

    inputs = Input(shape=(window_size, number_of_assays), name='input')

    assert(padding == "same")

    # Initial convolution
    x = BAC(inputs, conv_kernel_size, num_filters)

    # Perform multiple convolutional layers
    for i in range(num_convolutions):
        x = residual_block(x, conv_kernel_size, num_filters)

    # Final convolution
    outputs = BAC(x, conv_kernel_size, number_of_assays)

    # construct the CNN
    model = Model(inputs=inputs, outputs=outputs)

    return model


def create_transcriptome_cnn(number_of_assays,
                             window_size,
                             num_filters,
                             conv_kernel_size,
                             num_convolutions,
                             padding,
                             number_of_outputs):

    inputs = Input(shape=(window_size, number_of_assays), name='input')

    assert(padding == "same")
    # should drop out whole tracks
    x = Dropout(DROPOUT_PROB,
                noise_shape=(tf.shape(inputs)[0], 1, inputs.shape[2]))(inputs)
    # Initial convolution
    x = Dropout(DROPOUT_PROB)(BAC(inputs, conv_kernel_size, num_filters))

    # Perform multiple convolutional layers
    for i in range(num_convolutions):
        x = residual_block(x, conv_kernel_size, num_filters)

    # Final convolution
    # x = BAC(x, conv_kernel_size, 1)
    # outputs = Cropping1D(cropping=window_size // 2)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(DROPOUT_PROB)(BatchNormalization()(x))
    x = Dense(128, activation='relu')(x)
    x = Dropout(DROPOUT_PROB)(BatchNormalization()(x))
    outputs = Dense(number_of_outputs, activation='linear')(x)

    # construct the CNN
    model = Model(inputs=inputs, outputs=outputs)

    return model


if __name__ == '__main__':

    epochs = 1000
    steps_per_epoch = 100

    run_name_prefix = sys.argv[1]
    framework = sys.argv[2]  # epigenome or transcriptome
    window_size = int(sys.argv[3])  # => Length of window / 100bp
    batch_size = int(sys.argv[4])
    num_filters = int(sys.argv[5])
    conv_kernel_size = 11
    num_convolutions = int(sys.argv[6])
    padding = 'same'
    masking_prob = float(sys.argv[7])
    loss = sys.argv[8]
    if(((loss == 'mse') or (loss == 'mle')) is False):
        print("Loss should be mse or mle", file=sys.stderr)
        sys.exit(-1)
    genome_wide_flag = sys.argv[9]

    run_name = (run_name_prefix + "_" + framework + "_" + str(window_size) +
                "_" + str(num_filters) + "_" + str(num_convolutions) +
                "_" + str(masking_prob) + "_" + loss + "_" + genome_wide_flag)

    # tf.disable_v2_behavior()
    # tf.compat.v1.keras.backend.get_session()
    # tf.compat.v1.enable_eager_execution()
    # tf.enable_eager_execution()  # This leads to Filling up shuffle buffer

    if(framework == "epigenome"):

        loss_function = custom_loss
        DataGenerator = EpigenomeGenerator
        model = create_epigenome_cnn(len(ASSAY_TYPES),
                                     window_size,
                                     num_filters,
                                     conv_kernel_size,
                                     num_convolutions,
                                     'same')
    elif(framework == "transcriptome"):

        if(loss == 'mse'):
            loss_function = 'mean_squared_error'
            number_of_outputs = 1
        elif(loss == 'mle'):
            loss_function = maximum_likelihood_loss
            number_of_outputs = 2
        DataGenerator = TranscriptomeGenerator
        model = create_transcriptome_cnn(len(ASSAY_TYPES),
                                         window_size,
                                         num_filters,
                                         conv_kernel_size,
                                         num_convolutions,
                                         'same',
                                         number_of_outputs)
    else:
        print("Invalid framework. Should be epigenome or transcriptome",
              file=sys.stderr)
        sys.exit(-2)

    training_generator = DataGenerator(window_size,
                                       batch_size,
                                       shuffle=True,
                                       mode="training" + genome_wide_flag,
                                       masking_probability=masking_prob)

    validation_generator = DataGenerator(window_size,
                                         batch_size,
                                         shuffle=True,
                                         mode="validation" + genome_wide_flag,
                                         masking_probability=masking_prob)

    checkpoint = ModelCheckpoint(run_name+"."+"model-{epoch:02d}.hdf5",
                                 verbose=0, save_best_only=False)

    lr_schedule = LearningRateScheduler(lr_scheduler)

    tqdm_keras = TqdmCallback(verbose=0)
    setattr(tqdm_keras, 'on_train_batch_begin', lambda x, y: None)
    setattr(tqdm_keras, 'on_train_batch_end', lambda x, y: None)
    setattr(tqdm_keras, 'on_test_begin', lambda x: None)
    setattr(tqdm_keras, 'on_test_end', lambda x: None)
    setattr(tqdm_keras, 'on_test_batch_begin', lambda x, y: None)
    setattr(tqdm_keras, 'on_test_batch_end', lambda x, y: None)

    model.compile(loss=loss_function,
                  optimizer=Adam(),
                  run_eagerly=False)

    print(model.summary())

    model.fit(x=training_generator,
              epochs=epochs,
              verbose=0,
              callbacks=[checkpoint, lr_schedule, tqdm_keras],
              validation_data=validation_generator,
              validation_steps=steps_per_epoch,
              steps_per_epoch=steps_per_epoch)

    os._exit(1)
