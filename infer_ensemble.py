from ensemble import create_mlp
import sys
import numpy as np
import os
import sys
from keras.models import Sequential
from keras.layers import Dense


if __name__ == '__main__':
	track = sys.argv[1]
	chr = sys.argv[2]
	model_file = sys.argv[3]
	output_dir = sys.argv[4]
	
	# prefix = "/scratch/sanjit/ENCODE_Imputation_Challenge/ensembling/chr7/"
	prefix = "/scratch/sanjit/ENCODE_Imputation_Challenge/"

	x_avocado = np.expand_dims( np.load(prefix+"Blind_Avocado/numpy_arrays/"+track+"."+str(chr)+".npy"), axis=1)
	x_baseline = np.expand_dims( np.load(prefix+"Blind_Baseline/numpy_arrays/"+track+"."+str(chr)+".npy"), axis=1)
	x_cnn = np.expand_dims( np.load(prefix+"Blind_CNN/final_cnn_blind_imputation/"+track+"."+str(chr)+".npy"), axis=1)
	x_markov = np.expand_dims( np.load(prefix+"Blind_Markov/numpy_arrays/"+track+"."+str(chr)+".npy"), axis=1)

	# x_avocado = np.expand_dims( np.load(prefix+"Avocado/"+track+"."+str(chr)+".npy"), axis=1)
	# x_baseline = np.expand_dims( np.load(prefix+"Baseline/"+track+"."+str(chr)+".npy"), axis=1)
	# x_cnn = np.expand_dims( np.load(prefix+"CNN_8_16_64_0.25_sf64_log_MSE_regression/"+track+"."+str(chr)+".npy"), axis=1)
	# x_markov = np.expand_dims( np.load(prefix+"Markov_log/"+track+"."+str(chr)+".npy"), axis=1)

	X_test = np.log1p( np.maximum(   np.concatenate((x_avocado, x_baseline, x_cnn, x_markov), axis=1)+2 , 0 ) )

	model = create_mlp(32,8)
	model.load_weights(model_file)
	y_pred = np.maximum(    np.expm1( model.predict(X_test) ) - 2 ,    0)

	np.save(prefix+output_dir+track+"."+str(chr)+".npy", y_pred)

	

