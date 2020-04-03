"""
Read, process, and batch bigWig format data for the ENCODE challenge
"""
from __future__ import division
from __future__ import print_function
import numpy as np
from os.path import join, isfile
from numba import njit
from keras.utils import Sequence
# from multiprocessing import Pool


SEQ_DIR = '/scratch/sanjit/ENCODE_Imputation_Challenge/2_April_2020/Data/genome'
BINNED_DATA_DIR = ('/scratch/sanjit/ENCODE_Imputation_Challenge/2_April_2020'
		   '/Data/Training_Data')

# VALIDATION_BINNED_DATA_DIR = ('/scratch/sanjit/ENCODE_Imputation_Challenge'
#                              '/Smaller_Data/chr7_Validation/')

NUM_CELL_TYPES = 51
NUM_ASSAY_TYPES = 35
ALLOWED_CHROMS = set(['chr{}'.format(k) for k in list(range(1, 23)) + ['X']])


@njit('float32[:, :, :](int64, int64, int64, int64[:, :], float32[:, :])')
def make_array(i, j, k, indices, arrays):
    to_return = np.full((i, j, k), np.nan, dtype=np.float32)
    for i in range(indices.shape[0]):
        to_return[indices[i, 0], indices[i, 1], :] = arrays[i, :]
    return to_return


class BinnedHandler(Sequence):

    def __init__(self, data_len, batch_size):
        self.data_len = data_len
        self.batch_size = batch_size
        self.data = {}
        self.chrom_lens = {}
        self.indices = {}
        for cell_type in range(1, NUM_CELL_TYPES + 1):
            for assay_type in range(1, NUM_ASSAY_TYPES + 1):
                for chrom in [str(k) for k in range(1, 23)] + ['X']:
                    fname = 'C{:02}M{:02}.chr{}.npy'.format(cell_type,
                                                            assay_type,
                                                            chrom)
                    fname = join(BINNED_DATA_DIR, fname)
                    if isfile(fname):
                        if chrom == '1':
                            print('Loading',
                                  fname.split('/')[-1].split('.')[0])
			# Caution: We are working with log10(-log10 p-values ?)
                        this_array = np.log1p( np.load(fname) ) 
                        # np.log1p(np.load(fname))
                        if chrom not in self.data:
                            self.data[chrom] = {}
                            self.indices[chrom] = None
                            self.chrom_lens[chrom] = this_array.shape[0]
                        self.data[chrom][(cell_type, assay_type)] = this_array
        print('...Stacking arrays')
        for chrom in self.data.keys():
            indices, array = zip(*self.data[chrom].items())
            self.data[chrom] = np.vstack(array)
            self.indices[chrom] = np.array(indices)
        self.chrom_list, self.tot_len_list = zip(*self.chrom_lens.items())
        self.tot_len_list = np.array(self.tot_len_list) - self.data_len
        self.tot_len_list = np.cumsum(self.tot_len_list)
        # TODO: the following line will introduce some edge effects on the
        # final chromosome.
        self.length = self.tot_len_list[-1] // self.batch_size
        self.idx_map = np.arange(self.tot_len_list[-1])
        print(self.chrom_lens)

    def idx_to_chrom_and_start(self, idx):
        chr_idx = np.where(self.tot_len_list > idx)[0][0]
        chrom = self.chrom_list[chr_idx]
        start = idx if chr_idx == 0 else idx - self.tot_len_list[chr_idx - 1]
        return chrom, start

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        batch = []
        for i in range(self.batch_size*idx, self.batch_size*(idx+1)):
            chrom, start = self.idx_to_chrom_and_start(self.idx_map[i])
            batch.append(self.load_data(chrom, start, start + self.data_len))
        return np.array(batch)

    def load_data(self, chrom, start, end):
        return make_array(NUM_CELL_TYPES,
                          NUM_ASSAY_TYPES,
                          end - start,
                          self.indices[chrom],
                          self.data[chrom][:, start:end])


class BinnedHandlerImputing(BinnedHandler):

    def __init__(self, data_len, seg_len, CT_exchangeability=True):
        BinnedHandler.__init__(self, data_len, 1)
        self.seg_len = seg_len
        self.CT_exchangeability = CT_exchangeability
        self.chrom_list, self.tot_len_list = zip(*self.chrom_lens.items())
        self.tot_len_list = np.array(self.tot_len_list, dtype=float)
        self.tot_len_list -= self.data_len // 2
        self.tot_len_list /= (self.seg_len - self.data_len + 1)
        self.tot_len_list = np.ceil(self.tot_len_list).astype(int)
        self.tot_len_list = np.cumsum(self.tot_len_list)

    def idx_to_chrom_and_start(self, idx):
        chrom, start = BinnedHandler.idx_to_chrom_and_start(self, idx)
        start *= (self.seg_len - self.data_len + 1)
        return chrom, start

    def __len__(self):
        return self.tot_len_list[-1]

    def __getitem__(self, idx):
        chrom, start = self.idx_to_chrom_and_start(idx)
        end = start + self.seg_len
        if end > self.data[chrom].shape[1]:
            data = np.full((self.data[chrom].shape[0], self.seg_len), -1,
                           dtype=np.float32)
            data_we_have = self.data[chrom][:, start:]
            data[:, :data_we_have.shape[1]] = data_we_have
        else:
            data = self.data[chrom][:, start:end]
        batch = make_array(NUM_CELL_TYPES,
                           NUM_ASSAY_TYPES,
                           self.seg_len,
                           self.indices[chrom],
                           data)
        batch = np.expand_dims(batch, axis=0)
        batch[np.isnan(batch)] = -1

        if(self.CT_exchangeability):
            return batch.transpose((0, 1, 3, 2))
        else:
            return batch.transpose((0, 2, 3, 1 ))


class BinnedHandlerTraining(BinnedHandler):

    def __init__(self,
                 network_width,
                 batch_size,
                 seg_len=None,
                 drop_prob=0.5,
                 CT_exchangeability=True):
        if seg_len is None:
            seg_len = network_width
        self.network_width = network_width
        self.drop_prob = drop_prob
        self.CT_exchangeability = CT_exchangeability
        BinnedHandler.__init__(self, seg_len, batch_size)
        self.idx_map = np.random.permutation(self.tot_len_list[-1])

    def __getitem__(self, idx):
        batch = BinnedHandler.__getitem__(self, idx)  # already pre-processed
        return create_exchangeable_training_data(batch,
                                                 drop_prob=self.drop_prob,
                                                 data_len=self.network_width,
                                  CT_exchangeability=self.CT_exchangeability)

    def on_epoch_end(self):
        self.idx_map = np.random.permutation(self.tot_len_list[-1])


class SeqHandler(object):

    def __init__(self, seq_len):
        self.dna = {}
        self.seq_len = seq_len
        print('...Reading DNA sequences now')
        for chrom in ['chr' + str(k) for k in range(1, 23)] + ['chrX']:
            fname = join(SEQ_DIR, chrom + '.npy')
            self.dna[chrom[3:]] = np.load(fname)

    def get_dna(self, indices):
        seq = []
        for i in indices:
            chrom, start = self.idx_to_chrom_and_start(self.idx_map[i])
            start = start * 25
            end = start + self.seq_len*25
            this_seq = self.dna[chrom][start:end]
            this_seq = np.pad(this_seq,
                              (0, self.seq_len*25 - this_seq.shape[0]),
                              'constant',
                              constant_values=4)
            to_append = np.zeros((4, self.seq_len*25))
            keep = this_seq < 4
            to_append[this_seq[keep], np.arange(self.seq_len*25)[keep]] = 1
            seq.append(to_append.flatten())
        return seq


class BinnedHandlerSeqTraining(BinnedHandlerTraining, SeqHandler):
    '''Includes sequence information'''

    def __init__(self,
                 network_width,
                 batch_size,
                 seg_len=None,
                 drop_prob=0.5,
                 CT_exchangeability=True):
        BinnedHandlerTraining.__init__(
            self, network_width, batch_size, seg_len, drop_prob, CT_exchangeability)
        SeqHandler.__init__(self, self.data_len)

    def __getitem__(self, idx):
        x, y = BinnedHandlerTraining.__getitem__(self, idx)
        seq = self.get_dna(range(self.batch_size*idx, self.batch_size*(idx+1)))
        x = x.reshape((self.batch_size, -1))
        x = np.hstack([x, seq])
        return x, y


class BinnedHandlerSeqImputing(BinnedHandlerImputing,
                               SeqHandler):

    def __init__(self, data_len, seg_len, CT_exchangeability=True):
        BinnedHandlerImputing.__init__(self, data_len, seg_len, CT_exchangeability)
        SeqHandler.__init__(self, seg_len)

    def __getitem__(self, idx):
        to_return = BinnedHandlerImputing.__getitem__(self, idx)
        seq = self.get_dna([idx])
        to_return = to_return.reshape((1, -1))
        to_return = np.hstack([to_return, seq])
        return to_return


def create_exchangeable_training_data(batch, drop_prob, data_len=None, CT_exchangeability=True):
    x, y = zip(
        *[create_exchangeable_training_obs(b, drop_prob, data_len, CT_exchangeability)
          for b in batch]
    )
    return np.array(x), np.array(y)


# The input to this function is a single observation from the batch
def create_exchangeable_training_obs(obs, drop_prob, data_len=None, CT_exchangeability=True):
    if data_len is None:
        data_len = obs.shape[-1]
    input_tensor = np.copy(obs)
    # The dimensions of this np.array are (n x m x l)
    start = data_len//2
    end = start + obs.shape[-1] - data_len + 1
    output = np.copy(input_tensor[:, :, start:end])
    output = np.swapaxes(output, 1, 2)
    output = output.flatten()

    # Randomly drop each of the NUM_CELL_TYPES x NUM_ASSAY_TYPES
    # experiments with probability drop_prob
    mask = np.random.uniform(size=(NUM_CELL_TYPES, NUM_ASSAY_TYPES))
    mask = mask <= drop_prob
    mask = np.tile(mask, [input_tensor.shape[-1], 1, 1])
    mask = mask.transpose((1, 2, 0))
    input_tensor[mask] = -1.0

    # In both, input_feature and output, replace all nans
    # (denoting missing entries) with -1s
    input_tensor[np.isnan(input_tensor)] = -1.0
    output[np.isnan(output)] = -1.0

    if(CT_exchangeability):
        # switch the m and l dimensions to obtain a (n x l x m) tensor
        input_tensor = np.swapaxes(input_tensor, 1, 2)
    else:
        # If assay-type exchangeability, then obtain a (m x l x n) tensor instead
        input_tensor = np.swapaxes(input_tensor, 1, 2)
        input_tensor = np.swapaxes(input_tensor, 0, 2) 

    return input_tensor, output
