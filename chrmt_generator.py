# This script loads epigenetic data into memory
# It does this for multiple ChIP-seq assays
# and also for DNA Methylation (?)
# It does so for multiple cell types
# It then creates a generator
# which contains a masking function
import numpy as np
from os.path import join, isfile
from tensorflow.python.keras.utils.data_utils import Sequence
import sys
from random import randrange


DATA_FOLDER = '/scratch/sanjit/ENCODE_Imputation_Challenge/2_April_2020/Data/100bp_12_7_Data_20_July_2020' 

CELL_TYPES = ["T01", "T05"]

ASSAY_TYPES = ["A02", "A03", "A04", "A05", "A06", "A07"]

training_chroms = ["chr16"]

MASK_VALUE = 0


def preprocess_data(data):

	return np.log1p(data)


def create_masked(x):

	# dimensions are window_size x len(ASSAY_TYPES)
	# we mask out some portions by setting them to mask_value
	for i in range(x.shape[0]):
		if(np.random.uniform(low=0.0, high=1.0) < 0.95):
			x[i, :] = MASK_VALUE

	return x


class DataGenerator(Sequence):
	
	def __init__(self, window_size, batch_size, shuffle=True, mode=''):

		self.window_size = window_size
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.mode = mode

		self.data = {}
		self.chrom_lens = {}

		for chrom in training_chroms:
			for cell_type in CELL_TYPES:
				data = []
				for assay_type in ASSAY_TYPES:
					fname = cell_type+""+assay_type+"."+chrom+".npy"
					fname = DATA_FOLDER+"/"+fname
					if(isfile(fname)):
						print("Loading", fname, file=sys.stderr)
						current_data = np.load(fname)
						current_data = preprocess_data(current_data)
						data.append(current_data)
					else:
						print(assay_type, "missing in", cell_type, chrom)
						sys.exit(-1)

				if(chrom not in self.data):
					self.data[chrom] = {}
					self.chrom_lens[chrom] = current_data.shape[0]
				data = np.vstack(data)	# concatenate all assay types
				self.data[chrom][cell_type] = data


		# Now we need a way to randomly sample from the genome
		# For this we need chromosome lengths
		# We build a mapping from indexes to (chrom, position_in_chrom)
		self.chrom_list, self.tot_len_list = zip(*self.chrom_lens.items())
		self.tot_len_list = np.array(self.tot_len_list)
		self.tot_len_list = np.cumsum(self.tot_len_list)
		self.idxs = np.arange(self.tot_len_list[-1])


	def __len__(self):

		return self.tot_len_list[-1] // self.batch_size


	def on_epoch_end(self):

		if self.shuffle:
			np.random.shuffle(self.idxs)


	def idx_to_chrom_and_start(self, idx):

		chr_idx = np.where(self.tot_len_list > idx)[0][0]
		chrom = self.chrom_list[chr_idx]
		start = idx if chr_idx == 0 else idx - self.tot_len_list[chr_idx - 1]
		return chrom, start


	def __getitem__(self, batch_number):

		X = np.zeros((self.batch_size, self.window_size, len(ASSAY_TYPES)))
		Y = np.zeros((self.batch_size, self.window_size, len(ASSAY_TYPES)))

		for i in range(self.batch_size):
			idx = self.idxs[batch_number * self.batch_size + i]
			chrom, start = self.idx_to_chrom_and_start(idx)
			end = start + self.window_size

			if( (start < 1000) or (end > self.chrom_lens[chrom] + 1000) ):
				# We are too close to the edges of the chromosome
				# So we create a dummy point with all 0s
				# Since X and Y are aleady 0s, we do nothing
				print("We are too close to the edge!", file=sys.stderr)
				pass
			else:
				if(self.mode == 'train'):
					# Randomly sample a cell type
					random_cell_type_index = randrange(len(CELL_TYPES))
				else:
					random_cell_type_index = 0 # Fix cell type for testing
				random_cell_type = CELL_TYPES[random_cell_type_index]

				y = np.transpose(self.data[chrom][random_cell_type][:,start:end])
				x = create_masked(y)

				# print(x.shape, y.shape)

				X[i] = x
				Y[i] = y
				
		return X, Y
