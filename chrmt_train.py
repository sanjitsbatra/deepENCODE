import sys
import os
from chrmt_generator import DataGenerator, ASSAY_TYPES, MASK_VALUE
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Model
# from tensorflow.keras.layers import BatchNormalization, Input, add
from tensorflow.keras.layers import Conv1D, Input
# from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten
# from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from keras import backend as K
# from tqdm import tqdm


def lr_scheduler(epoch):
    if epoch < 10:
        return 2e-3
    elif epoch < 90:
        return 1e-3
    else:
        return 5e-4


# Compute an MSE loss only at those positions that are 0
def custom_loss(yTrue, yPred):
    print(yPred)

    masked_indices = K.tf.where(K.tf.equal(yTrue, MASK_VALUE),
                                K.tf.ones_like(yTrue),
                                K.tf.zeros_like(yTrue))

    loss = K.square(yTrue-yPred) * masked_indices
    return K.sum(loss, axis=-1) / K.sum(masked_indices, axis=-1)


def create_cnn(number_of_assays,
               window_size,
               num_filters,
               conv_kernel_size,
               num_convolutions,
               padding):
    inputs = Input(shape=(window_size, number_of_assays), name='input')

    print("Initial", inputs)

    x = inputs
    # Initial convolution
    x = Conv1D(kernel_size=conv_kernel_size,
               filters=num_filters,
               padding=padding)(x)

    print("After initial", x)

    # Perform multiple convolutional layers
    for i in range(num_convolutions):
        x = Conv1D(kernel_size=conv_kernel_size,
                   filters=num_filters,
                   padding=padding)(x)

    print("After multiple", x)

    # Final convolution
    outputs = Conv1D(kernel_size=conv_kernel_size,
                     filters=number_of_assays,
                     padding=padding)(x)

    print("After final", outputs)
    K.print_tensor(outputs, message='After final')

    # construct the CNN
    model = Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == '__main__':

    epochs = 100
    steps_per_epoch = 100

    run_name_prefix = sys.argv[1]
    window_size = int(sys.argv[2])  # => Length of window / 100bp
    batch_size = int(sys.argv[3])
    num_filters = int(sys.argv[4])
    conv_kernel_size = 7
    num_convolutions = int(sys.argv[5])
    padding = 'same'

    run_name = (run_name_prefix + "_" + str(window_size) +
                "_" + str(num_filters) + "_" + str(num_convolutions))

    # tensorflow.enable_eager_execution()

    train_generator = DataGenerator(window_size,
                                    batch_size,
                                    shuffle=True,
                                    mode='train')

    model = create_cnn(len(ASSAY_TYPES),
                       window_size,
                       num_filters,
                       conv_kernel_size,
                       num_convolutions,
                       'same')
    checkpoint = ModelCheckpoint(run_name+"."+"model-{epoch:02d}.hdf5",
                                 verbose=0, save_best_only=False)
    lr_schedule = LearningRateScheduler(lr_scheduler)
    model.compile(loss='mse', optimizer=Adam(clipnorm=1.), run_eagerly=True)
    print(model.summary())

    model.fit_generator(train_generator,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        callbacks=[checkpoint, lr_schedule],
                        use_multiprocessing=False,
                        verbose=2)

    os._exit(1)
