import sys, os
import numpy as np
import tensorflow, keras
from chrmt.generator import DataGenerator
from keras.models import load_model
from utils import lr_scheduler, clean_up_models
from keras.callbacks import LearningRateScheduler


def lr_scheduler(epoch):
	if epoch < 10:
		return 2e-3
	elif epoch < 90:
		return 1e-3
	else:
		return 5e-4


def BACUnit(conv_kernel_size, num_filters, padding):
	def f(input_node):
		bn = BatchNormalization()(input_node)
		act = Activation('relu')(bn)
		output_node = Conv1D(kernel_size=conv_kernel_size, 
							filters=num_filters, 
							padding=padding)(act)
		return output_node
	return f


def ResidualUnit(conv_kernel_size, num_filters, padding):
	def  f(input_node):
		bac1 = BACUnit(conv_kernel_size, num_filters, padding)(input_node)
		bac2 = BACUnit(conv_kernel_size, num_filters, padding)(bac1)
		if(padding == 'valid'):
			input_node = Cropping1D(conv_kernel_size - 1)(input_node)
		output_node = add([input_node, bac2]) 
		return output_node
	return f


def create_cnn(window_size, 
			number_of_channels, 
			num_filters, 
			conv_kernel_size, 
			num_convolutions, 
			padding):
	inputs = Input(shape=(window_size, number_of_channels), name='input') 

	# Initial convolution
	x = Conv1D(kernel_size=conv_kernel_size, 
			filters=num_filters, 
			padding=padding)(inputs)

	# Perform multiple convolutional layers
	for i in range(num_convolutions):
		x = ResidualUnit(conv_kernel_size, num_filters, padding)(x)

	output = x

	# construct the CNN
	model = Model(inputs=inputs, outputs=output)
	return model


if __name__ == '__main__':

	epochs = 100     
	steps_per_epoch = 100

	run_name_prefix = sys.argv[1]
	window_size = int(sys.argv[2]) # => Length of window / 100bp 
	batch_size = int(sys.argv[3])
	num_filters = int(sys.argv[4])
	conv_kernel_size = 11
	num_convolutions = 2
	padding = 'same'

	run_name = run_name_prefix+"_"+str(window_size)+"_"+str(num_filters)

	train_generator = DataGenerator(window_size, 
							batch_size, 
							shuffle=True, 
							mode='train')

	model = create_cnn(window_size, 
					len(ASSAY_TYPES), 
					num_filters, 
					conv_kernel_size, 
					num_convolutions, 
					'same')
	lr_schedule = LearningRateScheduler(lr_scheduler)
	model.compile(loss='mse', optimizer='adam')
	print(model.summary())

	model.fit_generator(train_generator, 
						epochs=epochs
						steps_per_epoch=steps_per_epoch,
						callbacks=[checkpoint, lr_schedule, csv_logger],
						use_multiprocessing=False, 
						verbose=2)

	os._exit(1)

