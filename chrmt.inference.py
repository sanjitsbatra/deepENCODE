import sys, os
import numpy as np
import tensorflow, keras
from chrmt.generator import DataGenerator
import h5py
import pandas as pd
from keras.models import load_model
from utils import lr_scheduler, clean_up_models
from keras.callbacks import LearningRateScheduler


if __name__ == '__main__':

	run_name_prefix = sys.argv[1]
	window_size = int(sys.argv[2]) # => Length of window / 100bp 
	batch_size = int(sys.argv[3])
	num_filters = int(sys.argv[4])

	run_name = run_name_prefix+"_"+str(window_size)+"_"+str(num_filters)

	test_generator = DataGenerator(window_size, 
							batch_size, 
							shuffle=False, 
							mode='test')

	final_model = load_model()
	print(model.summary())

	for i in tqdm(range(len(test_generator)):

		X_ref, Y = gen_ref.__getitem__(i)
		X_alt, Y = gen_alt.__getitem__(i)

		score_ref[i * args.batch_size:(i + 1) * args.batch_size] = model.predict(X_ref).mean(axis=(1, 2))
		score_alt[i * args.batch_size:(i + 1) * args.batch_size] = model.predict(X_alt).mean(axis=(1, 2))

	os._exit(1)

