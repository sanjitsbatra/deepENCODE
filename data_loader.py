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


# For Mac
# SEQ_DIR = '/Users/sbatra/Downloads/EIC/24_June_2020/genome'

# For Training
# BINNED_DATA_DIR = ('/Users/sbatra/Downloads/EIC/24_June_2020'
#              '/Training_Data_2')

# For Predicting
# BINNED_DATA_DIR = ('/Users/sbatra/Downloads/EIC/24_June_2020'
#            '/Testing_Data')

# GENE_EXPRESSION_DATA = ('/Users/sbatra/Downloads/EIC/24_June_2020'
#             '/GENE_EXPRESSION.NORMALIZED.tsv')


### For Clytius
SEQ_DIR = '/scratch/sanjit/ENCODE_Imputation_Challenge/2_April_2020/Data/genome'

# For Training
BINNED_DATA_DIR = ('/scratch/sanjit/ENCODE_Imputation_Challenge/2_April_2020'
                    '/Data/Training_Data')

# For Predicting
# BINNED_DATA_DIR = ('/scratch/sanjit/ENCODE_Imputation_Challenge/2_April_2020'
#                    '/Data/Testing_Data')

GENE_EXPRESSION_DATA = ('/scratch/sanjit/ENCODE_Imputation_Challenge/2_April_2020'
            '/Data/Gene_Expression/GENE_EXPRESSION.NORMALIZED.tsv')


# DECREASING NUM_CELL_TYPES TO LOWER THAN 12 LEADS TO MALLOC ERROR
# Also T12 data seems to cause the same error!
# These errors should now be fixed by the 28 June 2020 correction in the
# make_input_for_regression function 
NUM_CELL_TYPES = 12 
NUM_ASSAY_TYPES = 7

ALLOWED_CHROMS = set(['chr{}'.format(k) for k in list(range(1, 23)) + ['X']])


# Reverse complement for negative strand genes
def reverse_complement(seq):
    rev_seq = seq[::-1]
    rev_comp_dict = {0:3, 1:2, 2:1, 3:0, 4:4}
    rev_comp = [rev_comp_dict[c] for c in rev_seq]
    return np.asarray(rev_comp)


# For converting p-values into classes for Classification
def convert_to_classes(input_array, threshold):
    return np.where(input_array > threshold, np.float32(1.0), np.float32(0.0))
    # return np.asarray([1 if x > threshold else 0 for x in input_array])


@njit('float32[:, :, :](int64, int64, int64, int64[:, :], float64[:, :])')
def make_input_for_regression(num_cell_types, num_assay_types, 
                              twice_window_size, indices, data):
  
    to_return = np.full((num_cell_types, num_assay_types, twice_window_size), 
                         np.nan, dtype=np.float32) 
    
    for ii in range(indices.shape[0]):
        if ( (indices[ii, 0] - 1 >= num_cell_types) or
             (indices[ii, 1] - 1>= num_assay_types) ):
            print("The self indices are more than the size of the array!")
            continue

        to_return[indices[ii, 0] - 1, indices[ii, 1] - 1, :] = data[ii, :]
        # 28 June 2020
        # Added -1 to the above line because 12 was an index causing segfaults!

    return to_return


# If we ever switch back to this, clone corrections from above function
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

        # For training
        chrom_list =  ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 
                       'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr21']

        # For testing
        # chrom_list = ['chr13', 'chr14', 'chr15', 'chr16'] #, 'chr17', 'chr18',
        #               'chr19', 'chr20', 'chr22', 'chrX']

        for cell_type in range(1, NUM_CELL_TYPES + 1):
            for assay_type in range(1, NUM_ASSAY_TYPES + 1):
                for chrom in [str(k) for k in range(1, 23)] + ['X']:
                    fname = 'T{:02}A{:02}.chr{}.npy'.format(cell_type,
                                                            assay_type,
                                                            chrom)
                    fname = join(BINNED_DATA_DIR, fname)
                    if isfile(fname):
                        if "chr"+chrom not in chrom_list:
                            continue
                        else:
                            print("Loading chr", chrom,  
                                  fname.split('/')[-1].split('.')[0])

                        this_array = np.load(fname) 
			
                        # For keeping the epigenetic features continuous
                        # Caution: We are working with log10(-log10 p-values ?)
                        # this_array = np.log1p(this_array)

                        # Instead of taking the log10 of -log10 p-values
                        # We can perform arcsinh(-log10 p-values - 3)
                        this_array = np.arcsinh(this_array - 3)

                        # For binarizing the epigenetic features
                        # Convert -log10(p-values) into classes
                        # this_array = convert_to_classes(this_array, 3)

                        if chrom not in self.data:
                            self.data[chrom] = {}
                            self.indices[chrom] = None
                            self.chrom_lens[chrom] = this_array.shape[0]
                        self.data[chrom][(cell_type, assay_type)] = this_array
        print('...Stacking arrays')

        for chrom in self.data.keys():
            print(chrom, "building arrays")

            indices, array = zip(*self.data[chrom].items())

            # shape is: (cell_types*assay_types, chrom_length / 100)
            self.data[chrom] = np.vstack(array)
            # print("Shape of self.data", chrom, "is ", self.data[chrom].shape)

            # shape is: (cell_types*assay_types, 2)
            self.indices[chrom] = np.array(indices)
            # print("Shape of self.indices", chrom, "is",
            #        self.indices[chrom].shape)     
            # print("self.indices", self.indices[chrom])

        print("Beginning to parse Gene Expression now")

        # Instead of using idx to generate training data 
        # by sampling random genomic loci, we can now use a list of genes
        self.gene_position = {}
        self.gene_expression = {}

        f_gene_expression = open(GENE_EXPRESSION_DATA, 'r')
        line_number = 0
        for line in f_gene_expression:
            line_number += 1
            vec = line.rstrip("\n").split("\t")

            chrom_name = vec[0][3:] # remove the chr prefix 
         
            # Load only chromosomes whose epigenetics have been loaded
            if("chr"+chrom_name not in chrom_list):
                # print("Skipping chr"+chrom_name+" gene expression")
                continue

            # BUG discovered: 11 August 2020
            # BUG fixed: 23 August 2020 by incoporating strand everywhere
            # Incorporate strand information to distinguish TSS and TTS         	
            strand = vec[3]
            if(strand == "+"):
                strand = 1
                tss = int( int(vec[1]) / 100 )  # work at 100bp resolution
            elif(strand == "-"):
                strand = -1
                tss = int( int(vec[2]) / 100 )    
            else:
                print("Something went wrong with strand!")
                sys.exit(-1)

            gene_name = vec[5].split(".")[0]
            # gene_length = int( ( int(vec[2]) - int(vec[1]) ) / 100 )

            # Initialize a gene expression vector containing NUM_CELL_TYPE
            # entries, with the cell types for which we don't have 
            # gene expression values, always storing -1, for each gene
            # This is being done so that the custom loss function ignores
            # these -1 values while computing the MSE loss; while still 
            # being able to transfer learn from the EIC network output

            self.gene_position[gene_name] = (chrom_name, tss, strand)
            self.gene_expression[gene_name] = np.full((NUM_CELL_TYPES, 1), 
                                                -1000.0, dtype=np.float32)

            for col_i in range(6, len(vec)):
                # print(col_i-6, vec[col_i])
                self.gene_expression[gene_name][col_i-6] = float(vec[col_i])

        self.gene_names = self.gene_expression.keys()

    def __len__(self):
        return len(self.gene_names)

    # This function needs to generate random genes
    def __getitem__(self, idx):
        batch = []

        # randomly sample genes
        genes = random.sample(self.gene_names, self.batch_size)

        for gene in genes:       
            chrom, tss, strand = self.gene_position[gene]
            gene_expression = self.gene_expression[gene]

            # TSS-hopper: TSS +- \sigma and expression +- \delta


            # save the gene_names, (cell_type x assay_type x 2*window_wize) 
            # and gene expression values as a list
            batch.append([gene, 
                          self.load_gene_data(chrom, tss, strand, 
                                              self.window_size),
                          gene_expression])
        
        # print("Batch created", batch[0][0])
        return batch

    def load_gene_data(self, chrom, tss, strand, window_size):
        if(strand == 1):
            data = self.data[chrom][:, tss-window_size:tss+window_size]
        elif(strand == -1):
            data = self.data[chrom][:, tss+window_size-1:tss-window_size-1:-1]
        else:
            print("Something went wrong with the strand information")
            sys.exit(-1)

        return make_input_for_regression(NUM_CELL_TYPES,
                          NUM_ASSAY_TYPES,
                          2*window_size,
                          self.indices[chrom],
                          data)


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
            chrom, tss, strand = self.gene_position[gene]
            start = (tss - self.window_size) * 100
            end = (tss + self.window_size) * 100
            this_seq = self.dna[chrom][max(start, 0):min(end, 
                                                self.chrom_lens[chrom]*100)]

            if(strand == 1):
                pass
            elif(strand == -1):
                this_seq = reverse_complement(this_seq)
            else:
                print("Something is wrong with the strand")
                sys.exit(-1)

            # print("Shape of this_seq before padding", len(this_seq))
            # Pad the input
            this_seq = np.pad(this_seq,
                              (max(0, 0-start), 
                               max(0, end - self.chrom_lens[chrom]*100)),
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
                 drop_prob=0.00,
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
                 drop_prob=0.00,
                 CT_exchangeability=True):

        BinnedHandlerTraining.__init__(
            self, window_size, batch_size, seg_len, drop_prob, 
            CT_exchangeability)
        
        SeqHandler.__init__(self) 

    def __getitem__(self, idx):
        gene_names, x, y = BinnedHandlerTraining.__getitem__(self, idx)


        # Create artificial epigenetic data ############
        # x = np.asarray([np.full((2*self.window_size, NUM_ASSAY_TYPES), i) 
        #                 for i in range(6, 6 + NUM_CELL_TYPES)])
        # x = np.repeat(x[np.newaxis,...], self.batch_size, axis=0)

        seq = self.get_dna(gene_names)
 
        # Display the input epigenetic data (mean-ed across the positions)
        # print(np.mean(x[4,:,:,:],axis=1))

        # print("shape of x", x.shape, "shape of each seq", seq[0].shape)        
        x = x.reshape((self.batch_size, -1))
        # print("shape of x after reshape", x.shape)  

        x = np.hstack([x, seq])
        # print("final shape of x with seq", x.shape)
        # print("y shape", np.squeeze(np.asarray(y)).shape)
        return x, np.squeeze(np.asarray(y)) # TODO make this faster


class BinnedHandlerPredicting(BinnedHandler):

    def __init__(self,
                 window_size,
                 batch_size, 
                 seg_len=None,
                 drop_prob=0.00, 
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
                 seg_len=None, 
                 drop_prob=0.00,
                 CT_exchangeability=True):

        BinnedHandlerPredicting.__init__(
            self, window_size, batch_size, seg_len, drop_prob, 
            CT_exchangeability)

        SeqHandler.__init__(self)

    def __getitem__(self, idx):
        gene_names, x, y = BinnedHandlerPredicting.__getitem__(self, idx)        

        # Create artificial epigenetic data ############
        # x = np.asarray([np.full((2*self.window_size, NUM_ASSAY_TYPES), i) 
        #                 for i in range(6, 6 + NUM_CELL_TYPES)])
        # x = np.repeat(x[np.newaxis,...], self.batch_size, axis=0)

        seq = self.get_dna(gene_names)
        
        x = x.reshape((self.batch_size, -1))
        x = np.hstack([x, seq])

        return x, np.squeeze(np.asarray(y))


def create_exchangeable_training_data(batch_of_inputs, drop_prob, window_size,
                                      CT_exchangeability=True):
    input_data = ([create_exchangeable_training_obs(x, 
                                                   drop_prob, 
                                                   window_size, 
                                                   CT_exchangeability)
                                                   for x in batch_of_inputs])
    
    return np.array(input_data)


# The input to this function is a single observation from the batch
def create_exchangeable_training_obs(obs, drop_prob, window_size, 
                                     CT_exchangeability=True):

    # The dimensions of this tensor are (n x m x l)
    input_tensor = np.copy(obs)

    # Randomly drop each of the NUM_CELL_TYPES x NUM_ASSAY_TYPES
    # experiments with probability drop_prob
    # mask = np.random.uniform(size=(NUM_CELL_TYPES, NUM_ASSAY_TYPES))
    # mask = mask <= drop_prob
    # mask = np.tile(mask, [input_tensor.shape[-1], 1, 1])
    # mask = mask.transpose((1, 2, 0))
    # print("Indexing into mask")
    # input_tensor[mask] = -1000

    # In both, input_feature and output, replace all nans
    # (denoting missing entries) with -1s
    input_tensor[np.isnan(input_tensor)] = -1000.0

    if(CT_exchangeability):
        # switch the m and l dimensions to obtain a (n x l x m) tensor
        input_tensor = np.swapaxes(input_tensor, 1, 2)
    else:
        # If assay-type exchangeability, then obtain a (m x l x n) tensor
        input_tensor = np.swapaxes(input_tensor, 1, 2)
        input_tensor = np.swapaxes(input_tensor, 0, 2) 

    return input_tensor
