"""
Read, process, and batch bigWig format data for the ENCODE challenge
"""
from __future__ import division
from __future__ import print_function
import numpy as np
from os.path import join, isfile
from numba import njit
from keras.utils import Sequence
import random, sys
# from multiprocessing import Pool


SEQ_DIR = '/scratch/sanjit/ENCODE_Imputation_Challenge/2_April_2020/Data/genome'
BINNED_DATA_DIR = ('/scratch/sanjit/ENCODE_Imputation_Challenge/2_April_2020'
           '/Data/Training_Data')
GENE_EXPRESSION_DATA = ('/scratch/sanjit/ENCODE_Imputation_Challenge/2_April_2020'
            '/Data/Gene_Expression/GENE_EXPRESSION.NORMALIZED.tsv')

NUM_CELL_TYPES = 51
NUM_ASSAY_TYPES = 35
ALLOWED_CHROMS = set(['chr{}'.format(k) for k in list(range(1, 23)) + ['X']])


# For converting p-values into classes for Classification
def convert_to_classes(input_array, threshold):
    return np.where(input_array > threshold, 1, 0)
    # return np.asarray([1 if x > threshold else 0 for x in input_array])


@njit('float32[:, :, :](int64, int64, int64, int64[:, :], float32[:, :])')
def make_input_for_regression(i, j, k, indices, data):
    to_return = np.full((i, j, k), np.nan, dtype=np.float32) 
    
    for i in range(indices.shape[0]):
        to_return[indices[i, 0], indices[i, 1], :] = data[i, :]
    return to_return


@njit('int64[:, :, :](int64, int64, int64, int64[:, :], int64[:, :])')
def make_input_for_classification(i, j, k, indices, data):
    # Replacing np.nan by -1 in the second argument in the line below
    # is essential for preventing cross-entropy loss from diverging 6 May 2020
    to_return = np.full((i, j, k), -1, dtype=np.int64) 
    
    for i in range(indices.shape[0]):
        to_return[indices[i, 0], indices[i, 1], :] = data[i, :]
    return to_return


class BinnedHandler(Sequence):

    def __init__(self, window_size, batch_size):
        self.window_size = window_size
        self.batch_size = batch_size

        self.data = {}
        self.chrom_lens = {}
        self.indices = {}

        chrom_list = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr11', 'chr12', 'chr21']

        for cell_type in range(1, NUM_CELL_TYPES + 1):
            for assay_type in range(1, NUM_ASSAY_TYPES + 1):
                for chrom in [str(k) for k in range(1, 23)] + ['X']:
                    fname = 'C{:02}M{:02}.chr{}.npy'.format(cell_type,
                                                            assay_type,
                                                            chrom)
                    fname = join(BINNED_DATA_DIR, fname)
                    if isfile(fname):
                        if "chr"+chrom in chrom_list:
                            print("Loading chr", chrom,  
                                  fname.split('/')[-1].split('.')[0])

                        this_array = np.load(fname) 
			
                        # For Regression:
                        # Caution: We are working with log10(-log10 p-values ?)
                        this_array = np.log1p(this_array)

                        # For Classification:
                        # Convert -log10(p-values) into classes
                        # this_array = convert_to_classes(this_array, 3)

                        if chrom not in self.data:
                            self.data[chrom] = {}
                            self.indices[chrom] = None
                            self.chrom_lens[chrom] = this_array.shape[0]
                        self.data[chrom][(cell_type, assay_type)] = this_array
        print('...Stacking arrays')

        for chrom in self.data.keys():
            indices, array = zip(*self.data[chrom].items())

            # shape is: (chrom_length/25, cell_types*assay_types)
            self.data[chrom] = np.vstack(array)

            # shape is: (cell_types*assay_types, 1)
            self.indices[chrom] = np.array(indices)



        # # contains #chroms, chrom_lengths
        # self.chrom_list, self.tot_len_list = zip(*self.chrom_lens.items())


        # Instead of using idx to generate training data 
        # by sampling random genomic loci, we can now use a list of genes
        self.gene_position = {}
        self.gene_expression = {}
        cell_type_name = {}

        f_gene_expression = open(GENE_EXPRESSION_DATA, 'r')
        line_number = 0
        for line in f_gene_expression:
            line_number += 1
            vec = line.rstrip("\n").split("\t")

            if(line_number == 1):
                for col_i in range(6, len(vec)):
                    cell_type_name[col_i] = int(vec[col_i][1:3])
                continue

            chrom_name = vec[0][3:] # remove the chr prefix 
         
            # Load only chromosomes whose epigenetics have been loaded
            if("chr"+chrom_name not in chrom_list):
                print("Skipping chr"+chrom_name+" gene expression")
                continue
	
            tss = int( int(vec[1]) / 25 )  # work at 25bp resolution
            gene_name = vec[5].split(".")[0]
            # gene_length = int( ( int(vec[2]) - int(vec[1]) ) / 25 )

            # Initialize a gene expression vector containing NUM_CELL_TYPE
            # entries, with the cell types for which we don't have 
            # gene expression values, always storing -1, for each gene
            # This is being done so that the custom loss function ignores
            # these -1 values while computing the MSE loss; while still 
            # being able to transfer learn from the EIC network output
            self.gene_expression[gene_name] = np.full((NUM_CELL_TYPES, 1), 
                                                -1.0, dtype=np.float32)
            for col_i in range(6, len(vec)):
                self.gene_position[gene_name] = (chrom_name, tss)
                self.gene_expression[gene_name][cell_type_name[col_i]] = \
                                               float(vec[col_i])

        self.gene_names = self.gene_expression.keys()

        # ##################################################################
        # # What is this for?
        # self.data_len = data_len

        # self.tot_len_list = np.array(self.tot_len_list) - self.data_len
        # self.tot_len_list = np.cumsum(self.tot_len_list)
        # # TODO: the following line will introduce some edge effects on the
        # # final chromosome.
        # self.length = self.tot_len_list[-1] // self.batch_size
        # self.idx_map = np.arange(self.tot_len_list[-1])
        # print(self.chrom_lens)
        # ##################################################################

    # # idx is a genome-wide (chromosome-concatenated index) 
    # def idx_to_chrom_and_start(self, idx):
    #     chr_idx = np.where(self.tot_len_list > idx)[0][0]
    #     chrom = self.chrom_list[chr_idx]
    #     start = idx if chr_idx == 0 else idx - self.tot_len_list[chr_idx - 1]
    #     return chrom, start

    def __len__(self):
        return 128*self.batch_size

    # This function needs to generate random genes
    def __getitem__(self, idx):
        batch = []

        # randomly sample genes
        genes = random.sample(self.gene_names, self.batch_size)

        for gene in genes:       
            chrom, tss = self.gene_position[gene]
            gene_expression = self.gene_expression[gene]

            # save the gene_names, (cell_type x window_size x assay_type) 
            batch.append([gene, self.load_gene_data(chrom, tss, self.window_size),
                        gene_expression]) # and gene_expression values as list
        
        # print("Batch created", batch[0][0])
        return batch

    def load_gene_data(self, chrom, tss, window_size):
        return make_input_for_regression(NUM_CELL_TYPES,
                          NUM_ASSAY_TYPES,
                          2*window_size,
                          self.indices[chrom],
                          self.data[chrom][:, tss-window_size:tss+window_size])


# by passing object as argument, SeqHandler becomes an object of BinnedHandler
class SeqHandler(object):

    def __init__(self):
        self.dna = {}
        print('...Reading DNA sequences now')
        for chrom in ['chr' + str(k) for k in range(1, 23)] + ['chrX']:
            fname = join(SEQ_DIR, chrom + '.npy')

            # The value at each position is in {0, 1, 2, 3, 4}
            # corresponding to {'A', 'C', 'G', 'T', 'N'}
            self.dna[chrom[3:]] = np.load(fname)


    def get_dna(self, gene_names):
        seq = []
        for gene in gene_names:
            chrom, tss = self.gene_position[gene]
            start = (tss - self.window_size) * 25
            end = (tss + self.window_size) * 25
            this_seq = self.dna[chrom][max(start, 0):min(end, self.chrom_lens[chrom]*25)]

            # print("Shape of this_seq before padding", len(this_seq))
            # Pad the input
            this_seq = np.pad(this_seq,
                              (max(0, 0-start), max(0, end - self.chrom_lens[chrom]*25)),
                              'constant',
                              constant_values=4)
            # print("Shape of this_seq after padding", len(this_seq))

            output_seq = np.zeros((4, len(this_seq)))

            keep = this_seq < 4
            # print("Now we are indexing with keep")
            output_seq[this_seq[keep], np.arange(len(this_seq))[keep]] = 1

            # Output should be flattened because we unflatten in seq_module
            seq.append(output_seq.flatten())
        
        return seq


class BinnedHandlerTraining(BinnedHandler):

    def __init__(self,
                 window_size,
                 batch_size,
                 seg_len=None,
                 drop_prob=0.25,
                 CT_exchangeability=True):

        if seg_len is None:
            seg_len = window_size
        else:
            print("seg_len should not be used!")
            sys.exit(-1)
 
        self.window_size = window_size
        self.drop_prob = drop_prob
        self.CT_exchangeability = CT_exchangeability
 
        BinnedHandler.__init__(self, window_size, batch_size)

    def __getitem__(self, idx):
        batch = BinnedHandler.__getitem__(self, idx)
        
        gene_names = []
        inputs = []
        outputs = []
        
        # TODO: convert for to list comprehension
        for b in batch:
            gene_names.append(b[0])
            inputs.append(b[1])
            outputs.append(b[2])
        
        return gene_names, create_exchangeable_training_data(inputs,
                                                 drop_prob=self.drop_prob,
                                                 window_size=self.window_size,
                        CT_exchangeability=self.CT_exchangeability), outputs

    # def on_epoch_end(self):
    #     self.idx_map = np.random.permutation(self.tot_len_list[-1])


class BinnedHandlerSeqTraining(BinnedHandlerTraining, SeqHandler):
    '''Includes sequence information'''

    def __init__(self,
                 window_size,
                 batch_size,
                 seg_len=None,
                 drop_prob=0.25,
                 CT_exchangeability=True):

        BinnedHandlerTraining.__init__(
            self, window_size, batch_size, seg_len, drop_prob, 
            CT_exchangeability)
        
        SeqHandler.__init__(self) 

    def __getitem__(self, idx):
        gene_names, x, y = BinnedHandlerTraining.__getitem__(self, idx)

        seq = self.get_dna(gene_names)

        # print("shape of x", x.shape, "shape of each seq", seq[0].shape)        
        x = x.reshape((self.batch_size, -1))
        # print("shape of x after reshape", x.shape)  

        x = np.hstack([x, seq])
        # print("final shape of x with seq", x.shape)
        # print("y shape", np.squeeze(np.asarray(y)).shape)
        return x, np.squeeze(np.asarray(y)) # TODO fasten this


class BinnedHandlerPredicting(BinnedHandler):

    def __init__(self,
                 window_size,
                 batch_size, 
                 seg_len=None,
                 drop_prob=0.25, 
                 CT_exchangeability=True):

        if seg_len is None:
            seg_len = window_size
        else:
            print("seg_len should not be used!")
            sys.exit(-1)

        self.window_size = window_size
        self.drop_prob = drop_prob
        self.CT_exchangeability = CT_exchangeability

        BinnedHandler.__init__(self, window_size, 1)
        
    def __getitem__(self, idx):
        batch = BinnedHandler.__getitem__(self, idx)

        gene_names = []
        inputs = []
        outputs = []

        for b in batch:
            gene_names.append(b[0])
            inputs.append(b[1])
            outputs.append(b[2])

        return gene_names, create_exchangeable_training_data(inputs,
                                            drop_prob=self.drop_prob,
                                            window_size=self.window_size,
                        CT_exchangeability=self.CT_exchangeability), outputs


class BinnedHandlerSeqPredicting(BinnedHandlerPredicting, SeqHandler):

    def __init__(self, 
                 window_size, 
                 batch_size, 
                 drop_prob=0.25,
                 CT_exchangeability=True):

        BinnedHandlerPredicting.__init__(
            self, window_size, batch_size, seg_len, drop_prob, 
            CT_exchangeability)

        SeqHandler.__init__(self)

    def __getitem__(self, idx):
        gene_names, x, y = BinnedHandlerPredicting.__getitem__(self, idx)
        
        seq = self.get_dna(gene_names)
        
        x = x.reshape((self.batch_size, -1))
        x = np.hstack([x, seq])

        return x, np.squeeze(np.asarray(y))


def create_exchangeable_training_data(batch_of_inputs, drop_prob, window_size, CT_exchangeability=True):
    input_data = [create_exchangeable_training_obs(x, drop_prob, window_size, CT_exchangeability)
          for x in batch_of_inputs]
    
    return np.array(input_data)


# The input to this function is a single observation from the batch
def create_exchangeable_training_obs(obs, drop_prob, window_size, CT_exchangeability=True):

    # The dimensions of this tensor are (n x m x l)
    input_tensor = np.copy(obs)

    # Randomly drop each of the NUM_CELL_TYPES x NUM_ASSAY_TYPES
    # experiments with probability drop_prob
    mask = np.random.uniform(size=(NUM_CELL_TYPES, NUM_ASSAY_TYPES))
    mask = mask <= drop_prob
    mask = np.tile(mask, [input_tensor.shape[-1], 1, 1])
    mask = mask.transpose((1, 2, 0))
    # print("Indexing into mask")
    input_tensor[mask] = -1

    # In both, input_feature and output, replace all nans
    # (denoting missing entries) with -1s
    input_tensor[np.isnan(input_tensor)] = -1

    if(CT_exchangeability):
        # switch the m and l dimensions to obtain a (n x l x m) tensor
        input_tensor = np.swapaxes(input_tensor, 1, 2)
    else:
        # If assay-type exchangeability, then obtain a (m x l x n) tensor instead
        input_tensor = np.swapaxes(input_tensor, 1, 2)
        input_tensor = np.swapaxes(input_tensor, 0, 2) 

    return input_tensor
