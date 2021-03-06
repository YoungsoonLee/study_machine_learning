# reLU~!!!!!
import tensorflow as tf
import numpy as np
import input_data


mnist = input_data.read_data_sets('./mnist', one_hot=True)

learning_rate = 0.001
training_epochs = 25
batch_size = 100
display_step = 10

# tf graph input
X = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28*784
Y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes

# set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# construct model
activation = tf.nn.softmax(tf.matmul(X, W) + b) # softmax

# minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(activation), reduction_indices=1)) # cross entropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# init
init = tf.initialize_all_variables()

# launch
with tf.Session() as sess:
	sess.run(init)

	# Traning
	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(mnist.train.num_examples/batch_size)
		# Loop over all batches
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			# fit training using batch data
			sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
			# compute average loss
			avg_cost += sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys})/total_batch
		# display logs per epoch step
		#if epoch % display_step == 0:
		#	print("epoch: ", '%04d' % (epoch+1),  "cost=", "{:.9f}".format(avg_cost))
		if (epoch + 1) % display_step == 0:
			print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

	print("optimization finished")

	# test model
	correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(Y, 1))
	# calcurate accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
