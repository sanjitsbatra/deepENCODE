# This script collates the four methods' tracks into a big numpy array
import numpy as np
import os
import sys

def load_all_tracks(datapath):
	path = os.listdir(datapath)
	x = []
	for infile in path:
		s = os.path.join(datapath, infile)
		print s
		data = np.load(s)
		# print str(data.shape)
		x.append(data)  # np.concatenate((x, data), axis=0)
	return np.expand_dims(np.concatenate(x), axis=1)

if __name__ == '__main__':
	
	prefix = "/scratch/sanjit/ENCODE_Imputation_Challenge/ensembling/test_ensembling/"
	x_avocado = load_all_tracks(prefix+"Avocado/")
	x_baseline = load_all_tracks(prefix+"Baseline/")
	x_cnn = load_all_tracks(prefix+"CNN_8_16_64_0.25_sf64_log_MSE_regression/")
	x_markov = load_all_tracks(prefix+"Markov_log/")

	x_truth = load_all_tracks(prefix+"Truth/")

	X_train = np.concatenate((x_avocado, x_baseline, x_cnn, x_markov), axis=1)
	y_train = x_truth

	print str(x_avocado.shape)+" "+str(x_baseline.shape)+" "+str(x_cnn.shape)+" "+str(x_markov.shape)+" "+str(X_train.shape)+" "+str(y_train.shape)


	np.save("Ensembling.X_train",X_train)
	np.save("Ensembling.y_train",y_train)

