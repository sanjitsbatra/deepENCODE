import sys
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from chrmt_generator import EpigenomeGenerator, TranscriptomeGenerator
from chrmt_generator import ASSAY_TYPES, CELL_TYPES, MASK_VALUE, EPS
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import Conv1D, Input, add
from tensorflow.keras.layers import Cropping1D
from keras.models import load_model
# from tensorflow.keras.layers import Flatten, Dense
# from keras.losses import logcosh
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from keras import backend as K
from tqdm.keras import TqdmCallback
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def lr_scheduler(epoch):

    if epoch < 3:
        return 2e-3
    elif epoch < 90:
        return 1e-3
    else:
        return 5e-4


def clean_up_models(prefix, max_epoch_num):
    '''Deletes suboptimal models and removes epoch number from the best model'''
    for c in np.arange(1, max_epoch_num+1):
        try:
            os.rename(prefix+'-'+'{:03d}'.format(c)+'.hdf5', prefix+'.hdf5')
        except:
            continue
    return 0


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

    x_output = BatchNormalization()(x_input)
    x_output = Activation('relu')(x_output)
    x_output = Conv1D(kernel_size=conv_kernel_size,
                      filters=num_filters,
                      padding='same')(x_output)

    return x_output


def residual_block(x, conv_kernel_size, num_filters):

    x_1 = BAC(x, conv_kernel_size, num_filters)
    x_2 = BAC(x_1, conv_kernel_size, num_filters)
    output = add([x, x_2])

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

    # Initial convolution
    x = BAC(inputs, conv_kernel_size, num_filters)

    # Perform multiple convolutional layers
    for i in range(num_convolutions):
        x = residual_block(x, conv_kernel_size, num_filters)

    # Final convolution
    x = BAC(x, conv_kernel_size, 1)
    outputs = Cropping1D(cropping=window_size // 2)(x)

    # x = Flatten()(x)
    # x = Dense(64, activation='relu')(x)
    # outputs = Dense(number_of_outputs, activation='linear')(x)

    # construct the CNN
    model = Model(inputs=inputs, outputs=outputs)

    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name')
    parser.add_argument('--framework')
    parser.add_argument('--window_size', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--num_filters', type=int)
    args = parser.parse_args()

    run_name_prefix = args.run_name
    framework = args.framework 
    assert( (framework == "epigenome") or (framework == "transcriptome") )

    window_size = args.window_size  # => Length of window / RESOLUTION bp
    assert((window_size % 2) == 1)
    num_layers = args.num_layers
    num_filters = args.num_filters

    batch_size = 64
    conv_kernel_size = 5
    masking_prob = 0.0
    padding = 'same'

    loss = 'mse'
    if(((loss == 'mse') or (loss == 'mle')) is False):

        print("Loss should be mse or mle", file=sys.stderr)
        sys.exit(-1)
    else:
        assert(loss == 'mse')

    number_of_epochs = 100
    generate_dataframe = False

    genome_wide = int(False)

    run_name = (run_name_prefix + "_" + framework + "_" + str(genome_wide) + 
                "_" + str(window_size) + "_" + str(num_layers) +
                "_" + str(num_filters))

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
                                     num_layers,
                                     'same')
    elif(framework == "transcriptome"):

        if(loss == 'mse'):

            loss_function = 'mean_squared_error'
            number_of_outputs = 1

        elif(loss == 'mle'):

            loss_function = maximum_likelihood_loss
            number_of_outputs = 2

        DataGenerator = TranscriptomeGenerator

    else:

        print("Invalid framework. Should be epigenome or transcriptome",
              file=sys.stderr)
        sys.exit(-2)

    training_generator = DataGenerator(window_size,
                                       batch_size,
                                       shuffle=True,
                                       mode="training",
                                       masking_probability=masking_prob)

    validation_generator = DataGenerator(window_size,
                                         batch_size,
                                         shuffle=False,
                                         mode="validation",
                                         masking_probability=masking_prob)

    testing_generator = DataGenerator(window_size,
                                      batch_size,
                                      shuffle=False,
                                      mode="testing",
                                      masking_probability=masking_prob)

    tqdm_keras = TqdmCallback(verbose=0)
    setattr(tqdm_keras, 'on_train_batch_begin', lambda x, y: None)
    setattr(tqdm_keras, 'on_train_batch_end', lambda x, y: None)
    setattr(tqdm_keras, 'on_test_begin', lambda x: None)
    setattr(tqdm_keras, 'on_test_end', lambda x: None)
    setattr(tqdm_keras, 'on_test_batch_begin', lambda x, y: None)
    setattr(tqdm_keras, 'on_test_batch_end', lambda x, y: None)

    checkpoint = ModelCheckpoint("../../Models/" + run_name + "-" +
                                 "{epoch:03d}" + ".hdf5",
                                 verbose=0, save_best_only=True)

    lr_schedule = LearningRateScheduler(lr_scheduler)

    csv_logger = CSVLogger('../../Logs/' + run_name + '.csv', append=False)

    model = create_transcriptome_cnn(len(ASSAY_TYPES),
                                     window_size,
                                     num_filters,
                                     conv_kernel_size,
                                     num_layers,
                                     'same',
                                     number_of_outputs)

    model.compile(loss=loss_function,
                  optimizer='adam',
                  run_eagerly=False)

    print(model.summary())

    model.fit(training_generator,
                        epochs=number_of_epochs,
                        validation_data=validation_generator,
                        callbacks=[checkpoint, lr_schedule, csv_logger],
                        workers=0,
                        max_queue_size=0,
                        use_multiprocessing=False,
                        verbose=1)

    # Once the models have been trained, obtain a single best model
    clean_up_models('../../Models/' + run_name, number_of_epochs)

    # Once we have finished training the model
    # we want to compute to key sets of metrics
    # the first is the correlation of predictions across genes, for each cell type
    # the second is the correlation across cell types, for each gene
    # these two metrics then need to be summarized as figures, for each model
    trained_model = load_model('../../Models/' + run_name + '.hdf5')

    testing_generator_len = len(list(testing_generator))
    
    True_Expression = np.zeros((batch_size * testing_generator_len, 1))
    Predicted_Expression = np.zeros((batch_size * testing_generator_len, 1))
    metadata_list = []

    for test_i in range(testing_generator_len):

        x, yTrue, metadata = testing_generator.__getitem__(test_i)
        True_Expression[test_i * batch_size:(test_i + 1) * batch_size] = yTrue
        yPred = np.squeeze(trained_model.predict(x), axis=2)
        Predicted_Expression[test_i * batch_size:
                             (test_i + 1) * batch_size] = yPred
        metadata_list.append(metadata)
    metadata_list = np.reshape(np.asarray(metadata_list), (-1, 5))

    f_output = open('../../Logs/' + run_name + '.testing_metrics.tsv', 'w')
    for i in range(True_Expression.shape[0]):
        print(str(metadata_list[i, 0]) + "\t" +
              str(metadata_list[i, 1]) + "\t" +
              str(metadata_list[i, 2]) + "\t" +
              str(metadata_list[i, 3]) + "\t" +
              str(metadata_list[i, 4]) + "\t" +
              str(True_Expression[i, 0]) + "\t" +
              str(Predicted_Expression[i, 0]),
              file=f_output) 
    f_output.close()

    # Now we create a visualization of the training loss, validation loss
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(12, 12)
        
    epoch_number = []
    training_loss = []
    validation_loss = []
    f_loss = open('../../Logs/' + run_name + '.csv', 'r')
    line_number = 0
    for line in f_loss:
        line_number += 1
        if(line_number == 1):
            continue

        vec = line.rstrip("\n").split(",")
        epoch_number.append(int(vec[0]))
        training_loss.append(float(vec[1]))
        validation_loss.append(float(vec[3]))

    axs[0, 0].plot(epoch_number, training_loss, 'o-', color="#4daf4a", markersize=4, linewidth=3, label="training_loss")
    axs[0, 0].plot(epoch_number, validation_loss, 'o-', color="#8470FF", markersize=4, linewidth=3, label="validation_loss")

    axs[0, 0].set_xlim(-1, max(epoch_number) + 1)
    axs[0, 0].set_ylim(0, 1)
    axs[0, 0].set_xlabel("Number of epochs")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend(loc="upper center", fontsize=10, ncol=1)
    axs[0, 0].set_title("Loss vs epoch")

    # We also visualize the correlation across genes within each cell type
    cell_type_spearman = {}
    for cell_type in tqdm(CELL_TYPES):
        f_predictions = open('../../Logs/' + run_name + '.testing_metrics.tsv', 'r')
        yTrue = []
        yPred = []
        for line in f_predictions:
            vec = line.rstrip("\n").split("\t")
            if(vec[3]!=cell_type):
                continue
            else:
                yTrue.append(float(vec[5]))
                yPred.append(float(vec[6]))

        sc, sp = spearmanr(yTrue, yPred)
        cell_type_spearman[cell_type] = sc

        f_predictions.close()

        if(cell_type == "T13"):
            axs[0, 1].plot(yTrue, yPred, 'o', markersize=5, color="#FF1493")
            axs[0, 1].set_xlim(-1, 5)
            axs[0, 1].set_ylim(-1, 5)
            axs[0, 1].set_xlabel("True log10(TPM+1)")
            axs[0, 1].set_ylabel("Predicted log10(TPM+1)")
            axs[0, 1].set_title("HEK293T True vs Predicted gene expression")

    axs[1, 0].bar(CELL_TYPES, [cell_type_spearman[ct] for ct in CELL_TYPES], color=plt.cm.plasma(np.linspace(0.1, 0.9, len(CELL_TYPES))))
    axs[1, 0].set_title("Correlations across genes for different cell types")
    axs[1, 0].set_ylim(0, 1)

    # We then visualize the CDF of correlations across cell types for each gene
    f_predictions = open('../../Logs/' + run_name + '.testing_metrics.tsv', 'r')
    yTrue = {}
    yPred = {}
    for line in f_predictions:
        vec = line.rstrip("\n").split("\t")
        transcript = vec[4]
        if(transcript in yTrue):
            yTrue[transcript].append(float(vec[5]))
            yPred[transcript].append(float(vec[6]))
        else:
            yTrue[transcript] = [float(vec[5])]
            yPred[transcript] = [float(vec[6])]        

    f_predictions.close()             
    
    spearman_list = []
    CXCR4_spearman = -100
    TGFBR1_spearman = -100
    for transcript in yTrue:

        yT = yTrue[transcript]
        yP = yPred[transcript]
        sc, sp = spearmanr(yT, yP)

        spearman_list.append(sc)

        if(transcript == "ENST00000241393.3"):
            CXCR4_spearman = round(sc, 2)
        elif(transcript == "ENST00000374994.8"):
            TGFBR1_spearman = round(sc, 2)

    sns.set_style('whitegrid')
    sns.kdeplot(np.array(spearman_list), fill=True, alpha=0.5, linewidth=0.05, bw_adjust=0.2, color="#2ab0ff", ax=axs[1, 1])
    axs[1, 1].set_xlim(-1.1, 1.1)
    axs[1, 1].set_ylim(0, 2)
    axs[1, 1].set_title("Spearman across cell types for genes\n"+
                        "CXCR4 spearman = "+str(CXCR4_spearman)+
                        " TGFBR1 spearman = "+str(TGFBR1_spearman))
                        
    fig.savefig("../../Results/" + run_name + ".testing.pdf")

    os._exit(1)
