model = load_model

compute_loss(model, x):
	yPred = model.predict(x)
	loss = np.square(yPred[ct] - yTrue[ct]) + ?

@tf.function
def gradient_ascent_step(x):
	# x_var = tf.convert_to_tensor(x)
	with tf.GradientTape() as tape:
		tape.watch(x)
		loss = compute_loss(model, x) # set training=False?

	# Compute gradients
	grads = tape.gradient(loss, x)

	# Normalize gradients
	grads = tf.math.l2_normalize(grads)

	x += 1e-3 * grads # learning rate
	return loss, x 


for iteration in range(20):
	loss, x = gradient_ascent_step(x)
