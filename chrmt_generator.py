# This script loads epigenetic data into memory
# It does this for multiple ChIP-seq assays
# and also for DNA Methylation (?)
# It then also loads genome-wide RNA-seq data binned at 100bp resolution
# It does so for multiple cell types
# It then creates a generator
# which contains a masking function for the epigenetic data
# in order to perform masked-language modeling or supervised RNA-seq prediction
import numpy as np
from os.path import isfile
from tensorflow.python.keras.utils.data_utils import Sequence
import sys
from random import randrange
import pyranges as pr


EPS = 0.000001

DATA_FOLDER = '../Data/100bp_12_7_Data_20_July_2020'

TRANSCRIPTOME_DATA_FOLDER = "/scratch/sanjit/ENCODE_Imputation_Challenge/" \
                            "2_April_2020/Data/Gene_Expression/" \
                            "genome_wide_TPM_npy"

CELL_TYPES = ["T" + "{0:0=2d}".format(i) for i in range(1, 2)]  # 13

ASSAY_TYPES = ["A" + "{0:0=2d}".format(i) for i in range(2, 8)]

training_chroms = ["chr"+str(i) for i in range(1, 3, 2)]
validation_chroms = ["chr"+str(i) for i in range(2, 3, 2)]
testing_chroms = ["chr"+str(i) for i in range(2, 3, 2)]

MASK_VALUE = -10

EDGE_CUSHION = 1000  # corresponds to 100Kb from the edge of chromosomes


# We don't want to train in regions that are Blacklisted or have Gaps
Blacklisted_Regions = pr.read_bed('../Data/hg38.Blacklisted.bed', as_df=False)
Gap_Regions = pr.read_bed('../Data/hg38.Gaps.bed', as_df=False)


def check_region(chrom, start, end):
    # Since we are working at 100bp resolution
    start = int(start*1.0 * 100.0)
    end = int(end*1.0 * 100.0)

    if(len(Blacklisted_Regions[chrom, start:end]) +
       len(Gap_Regions[chrom, start:end]) > 0):
        return "bad"
    else:
        return "good"


def preprocess_epigenome(epigenome):

    return np.log1p(epigenome)


def create_masked(y, p):

    # dimensions are window_size x len(ASSAY_TYPES)
    # We mask out some positions in x and mask the opposite ones in y
    counter = 0
    x = np.copy(y)
    for i in range(x.shape[0]):
        if(np.random.uniform(low=0.0, high=1.0) < p):
            counter += 1
            x[i, :] = MASK_VALUE
        else:
            y[i, :] = MASK_VALUE

    '''
    # This can be used to debug how many entries are masked
    if(counter == 0):
        print("No entries have been masked", file=sys.stderr)
    elif(counter == x.shape[0]):
        print("All entries have been masked", file=sys.stderr)
    '''

    return x, y


class EpigenomeGenerator(Sequence):

    def __init__(self, window_size, batch_size,
                 shuffle=True, mode='', masking_probability=0.2):

        self.window_size = window_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.masking_probability = masking_probability

        self.epigenome = {}
        self.chrom_lens = {}
        self.transcriptome_pos = {}
        self.transcriptome_neg = {}

        if(self.mode == 'train'):
            self.chroms = training_chroms
        elif(self.mode == 'validation'):
            self.chroms = validation_chroms
        elif(self.mode == 'test'):
            self.chroms = testing_chroms

        for chrom in self.chroms:
            for cell_type in CELL_TYPES:
                epigenome = []
                for assay_type in ASSAY_TYPES:
                    f_name = cell_type+""+assay_type+"."+chrom+".npy"
                    f_name = DATA_FOLDER+"/"+f_name
                    if(isfile(f_name)):
                        print("Loading Epigenome data", f_name,
                              file=sys.stderr)
                        current_epigenome = np.load(f_name)
                        current_epigenome = preprocess_epigenome(
                                            current_epigenome)
                        epigenome.append(current_epigenome)
                    else:
                        print(assay_type, "missing in", cell_type, chrom)
                        sys.exit(-1)

                # Load transcriptome
                f_transcriptome_pos = cell_type + "_TPM." + chrom + ".+.npy"
                f_transcriptome_pos = (TRANSCRIPTOME_DATA_FOLDER + "/"
                                       + f_transcriptome_pos)

                f_transcriptome_neg = cell_type + "_TPM." + chrom + ".-.npy"
                f_transcriptome_neg = (TRANSCRIPTOME_DATA_FOLDER + "/"
                                       + f_transcriptome_neg)

                if(isfile(f_transcriptome_pos) and
                   isfile(f_transcriptome_neg)):
                    print("Loading Transcriptome data", f_transcriptome_pos,
                          f_transcriptome_neg, file=sys.stderr)
                    transcriptome_pos = np.load(f_transcriptome_pos)
                    transcriptome_neg = np.load(f_transcriptome_neg)
                else:
                    print("Transcriptome data missing", f_transcriptome_pos,
                          f_transcriptome_neg, file=sys.stderr)
                    sys.exit(-2)

                if(chrom not in self.epigenome):
                    self.epigenome[chrom] = {}
                    self.chrom_lens[chrom] = current_epigenome.shape[0]
                    self.transcriptome_pos[chrom] = {}
                    self.transcriptome_neg[chrom] = {}
                epigenome = np.vstack(epigenome)  # concatenate all assay types
                self.epigenome[chrom][cell_type] = epigenome
                self.transcriptome_pos[chrom][cell_type] = transcriptome_pos
                self.transcriptome_neg[chrom][cell_type] = transcriptome_neg

        # Now we need a way to randomly sample from the genome
        # For this we need chromosome lengths
        # We build a mapping from indexes to (chrom, position_in_chrom)
        self.chrom_list, self.tot_len_list = zip(*self.chrom_lens.items())
        self.tot_len_list = np.array(self.tot_len_list)
        self.tot_len_list = np.cumsum(self.tot_len_list)
        self.idxs = np.arange(self.tot_len_list[-1])

    # Apparently Keras doesn't call this at the end of every epoch!!!
    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.idxs)

    # So we have to call shuffle inside len which is called after every epoch!
    def __len__(self):

        if self.shuffle:
            np.random.shuffle(self.idxs)

        return self.tot_len_list[-1] // self.batch_size

    def idx_to_chrom_and_start(self, idx):

        chr_idx = np.where(self.tot_len_list > idx)[0][0]
        chrom = self.chrom_list[chr_idx]

        d = -1
        if(chr_idx == 0):
            d = idx
        else:
            d = idx - self.tot_len_list[chr_idx - 1]
        start = d

        return chrom, start

    def __getitem__(self, batch_number):

        X = np.zeros((self.batch_size, self.window_size, len(ASSAY_TYPES)))
        Y = np.zeros((self.batch_size, self.window_size, len(ASSAY_TYPES)))

        number_of_data_points = self.batch_size
        while(number_of_data_points > 0):
            random_idx = randrange(self.tot_len_list[-1])
            idx = self.idxs[random_idx]
            chrom, start = self.idx_to_chrom_and_start(idx)
            end = start + self.window_size

            # Very useful for debugging!
            # print("Batch Number", batch_number, chrom, start, end,
            #       file=sys.stderr)

            if((start < EDGE_CUSHION) or
               (end > self.chrom_lens[chrom] - EDGE_CUSHION)):
                # We are too close to the edges of the chromosome
                '''
                print("We are too close to the edge!",
                      batch_number, idx, chrom, start,
                      end, self.chrom_lens[chrom],
                      file=sys.stderr)
                '''
                continue
            # TODO: This slows down training significantly
            # elif(check_region(chrom, start, end) == "bad"):
                # The training data point either lies in
                # a Blacklisted Region or a Gap Region in hg38
                # So we create the i'th data point to be a dummy with all 0s
                # Since X and Y are aleady 0s, we do nothing
                '''
                print("Data point in Blacklisted or Gap region",
                      batch_number, idx, chrom, start, d,
                      end, self.chrom_lens[chrom],
                      file=sys.stderr)
                '''
                # continue
            else:
                if((self.mode == 'train') or (self.mode == 'validation')):
                    # Randomly sample a cell type
                    random_cell_type_index = randrange(len(CELL_TYPES))
                    # print("Sampled cell type", random_cell_type_index,
                    #       "for training", file=sys.stderr)
                else:
                    random_cell_type_index = 0  # Fix cell type for testing

                random_cell_type = CELL_TYPES[random_cell_type_index]

                # TODO: remove this transpose
                # TODO: add assert on size to make sure it's always consistent
                y = self.epigenome[chrom][random_cell_type][:, start:end]
                y = np.transpose(y)

                x_masked, y_masked = create_masked(y, self.masking_probability)
                # print(x_masked, y_masked)

                '''
                if(x_masked.shape[0] != self.window_size):
                    print("Found the wrong shape!",
                          chrom, start, end, x_masked.shape, y_masked.shape,
                          file=sys.stderr)
                '''

                X[number_of_data_points-1] = x_masked
                Y[number_of_data_points-1] = y_masked

                number_of_data_points -= 1

        return X, Y


class TranscriptomeGenerator(EpigenomeGenerator):

    def __init__(self, window_size, batch_size,
                 shuffle=True, mode='', masking_probability=0.,
                 cell_type=None):

        self.window_size = window_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.masking_probability = masking_probability
        self.cell_type = cell_type

        EpigenomeGenerator.__init__(self, window_size, batch_size,
                                    shuffle, mode, masking_probability)

    def __getitem__(self, batch_number):

        X = np.zeros((self.batch_size, self.window_size, len(ASSAY_TYPES)))
        Y = np.zeros((self.batch_size, 1))

        number_of_data_points = self.batch_size
        while(number_of_data_points > 0):
            random_idx = randrange(self.tot_len_list[-1])
            idx = self.idxs[random_idx]
            chrom, start = self.idx_to_chrom_and_start(idx)
            end = start + self.window_size

            # Very useful for debugging!
            # print("Batch Number", batch_number, chrom, start, end,
            #       file=sys.stderr)

            if((start < EDGE_CUSHION) or
               (end > self.chrom_lens[chrom] - EDGE_CUSHION)):
                # We are too close to the edges of the chromosome
                '''
                print("We are too close to the edge!",
                      batch_number, idx, chrom, start,
                      end, self.chrom_lens[chrom],
                      file=sys.stderr)
                '''
                continue
            # TODO: This slows down training significantly
            # elif(check_region(chrom, start, end) == "bad"):
                # The training data point either lies in
                # a Blacklisted Region or a Gap Region in hg38
                # So we create the i'th data point to be a dummy with all 0s
                # Since X and Y are aleady 0s, we do nothing
                '''
                print("Data point in Blacklisted or Gap region",
                      batch_number, idx, chrom, start, d,
                      end, self.chrom_lens[chrom],
                      file=sys.stderr)
                '''
                # continue
            else:
                if((self.mode == 'train') or (self.mode == 'validation')):
                    # Randomly sample a cell type
                    random_cell_type_index = randrange(len(CELL_TYPES))
                    # print("Sampled cell type", random_cell_type_index,
                    #       "for training", file=sys.stderr)
                else:
                    random_cell_type_index = 0  # Fix cell type for testing

                random_cell_type = CELL_TYPES[random_cell_type_index]

                x = self.epigenome[chrom][random_cell_type][:, start:end]
                x = np.transpose(x)

                # Flip a coin to choose the strand
                random_toss = randrange(1)
                if(random_toss == 1):
                    y = (self.transcriptome_pos[chrom]
                                               [random_cell_type]
                                               [int(1.0*(start+end)/2)])
                else:
                    x = x[::-1, :]
                    y = (self.transcriptome_neg[chrom]
                                               [random_cell_type]
                                               [int(1.0*(start+end)/2)])

                X[number_of_data_points-1] = x
                Y[number_of_data_points-1] = y

                number_of_data_points -= 1

        return X, Y
