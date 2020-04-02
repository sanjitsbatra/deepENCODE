# This script takes as input an Nx4 numpy array, where the columns are 
# Avocado Baseline CNN Markov
# And each row represents a 25bp genomic position in a particular track
# It also takes as input a Nx1 array containing the corresponding ground truth

import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from keras.optimizers import Adam
import math
import matplotlib.pyplot as plt
# import cv2
from sklearn.model_selection import train_test_split, StratifiedKFold
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Dropout, Convolution2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import np_utils
from random import randint


def create_mlp(layer_1_nodes=32, layer_2_nodes=8):
	model = Sequential()

	model.add(Dense(layer_1_nodes, input_dim=4, kernel_initializer='normal', 
					activation='relu'))
	model.add(Dense(layer_2_nodes, kernel_initializer='normal', 
					activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))

	# Compile model
	opt = Adam(lr=1e-3)
	model.compile(loss='mean_squared_error', optimizer=opt)
	# print str(model.summary())

	return model


def data_generator(x, y, batch_size):
	while True:
		random_indices = [randint(0, len(x)-1) for p in range(0, batch_size)]
		yield(np.array(x[random_indices,:]), np.array(y[random_indices]))

if __name__ == '__main__':

	# X_train is an Nx4 numpy array containing data from chr4 for all tracks
	# concatenated in the same order as in y_train
	X_train = np.load(sys.argv[1])
	y_train = np.load(sys.argv[2])

	# Log everything but relu because Avocado has negative values sometimes
	X_train = np.log1p(np.maximum( X_train, 0 ))
	y_train = np.log1p(y_train)

	# Earlier we had done
	# X_train = np.log1p(X_train+2)
	# y_train = np.log1p(yTrain+2)

	# num_folds = 2
	batch_size = 256
	num_epochs = 10

	steps_per_epoch = 200000
	validation_steps_per_epoch = 50000

	# folds = list(KFold(n_splits=num_folds, shuffle=True, random_state=None).split(X_train, y_train))

	# Due to memory, we can't create multiple folds in memory
	# Instead we have to create a random split into X_train and X_valid
	x_len = X_train.shape[0]
	train_idx = np.random.choice(x_len, int(0.8*x_len), replace=False)
	val_idx = np.setdiff1d(np.arange(x_len),train_idx)
	print str(train_idx.shape)+" "+str(val_idx.shape)+" "+str(x_len)+" "+str(train_idx.shape[0]+val_idx.shape[0])
	folds = [(train_idx, val_idx)] 

	for j, (train_idx, val_idx) in enumerate(folds):
		print str(len(train_idx))+" "+str(len(val_idx))

		X_train_cv = X_train[train_idx]
		y_train_cv = y_train[train_idx]
		X_valid_cv = X_train[val_idx]
		y_valid_cv= y_train[val_idx]

		# name_weights = "final_model_fold." + str(j) + ".weights.hdf5"
		model = create_mlp(32,8)
		mcp_save = ModelCheckpoint("Relued.fold-"+str(j)+"_{epoch:02d}-{val_loss:.2f}.hdf5", 
					save_best_only=False, 
					monitor='val_loss', 
					mode='min')
		callbacks = [mcp_save]
		# generator = data_generator(X_train_cv, y_train_cv, batch_size)
		model.fit_generator(data_generator(X_train_cv, y_train_cv, batch_size),
					steps_per_epoch=steps_per_epoch,  # len(X_train_cv)/batch_size,
					epochs=num_epochs,
					shuffle=True,
					verbose=1,
					validation_data = data_generator(X_valid_cv, y_valid_cv, batch_size),
					validation_steps=validation_steps_per_epoch,
					callbacks = callbacks)# ,
					# workers=4,
					# use_multiprocessing=True,
					# max_queue_size=100)

		# model.fit(
		# 		x=X_train_cv, 
		# 		y=y_train_cv, 
		# 		batch_size=batch_size, 
		# 		# steps_per_epoch=len(X_train_cv)/batch_size,
		# 		epochs=num_epochs,
		# 		shuffle=True,
		# 		verbose=1,
		# 		validation_data=(X_valid_cv, y_valid_cv),
		# 		# validation_steps=1000,
		# 		callbacks=callbacks)
		m = model.evaluate(X_valid_cv, y_valid_cv)
		print("MSE on Validation data for ",j,"th fold = ", m)


	sys.exit(1)
