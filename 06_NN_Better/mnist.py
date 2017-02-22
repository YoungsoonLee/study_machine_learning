import tensorflow as tf
import numpy as np
import input_data


mnist = input_data.read_data_sets('./mnist', one_hot=True)

learning_rate = 0.001
training_epochs = 25
batch_size = 100
display_step = 10

# print(mnist.test.images)
print(mnist.test.labels)

"""
# tf graph input
x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28*784
y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes

# Create Model

# set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# construct model
activation = tf.nn.softmax(tf.matmul(x, W) + b) # softmax

# minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation), reduction_indices=1)) # cross entropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
"""

""" NN """
# tf graph input
X = tf.placeholder("float", [None, 784])  # MNIST data input (img shape: 28*28)
Y = tf.placeholder("float", [None, 10])  # MNIST total classes

# store layers weight & bias
W1 = tf.Variable(tf.random_normal([784, 256]))
W2 = tf.Variable(tf.random_normal([256, 256]))
W3 = tf.Variable(tf.random_normal([256, 10]))

B1 = tf.Variable(tf.random_normal([256]))
B2 = tf.Variable(tf.random_normal([256]))
B3 = tf.Variable(tf.random_normal([10]))

# construct
L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), B2))  # hidden layer
activation = tf.add(tf.matmul(L2, W3), B3) # No need to use softman here

# define loss and potimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(activation, Y)) # softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # adam optimizer
""" end NN """


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
