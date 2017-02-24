import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])  # y_data is vector

# tf graph input
X = tf.placeholder('float', [None, 3])  # x1, x2 and 1 (1 is for bias), None mesn that we don't know how many rows
Y = tf.placeholder('float', [None, 3])  # A, B, C => 3 classes, None mesn that we don't know how many rows
# set model weights
W = tf.Variable(tf.zeros([3, 3]))  # 3x3 matrix (x_data * y_data)

# Construct model
hypothesis = tf.nn.softmax(tf.matmul(X, W)) # softmax

# Minimize error using cross entropy
learning_rate = 0.001 # affect to cost step

# Cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices=1))

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# launch the graph
with tf.Session() as sess:
	sess.run(init)

	for step in range(2001):
		sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
		if step % 200 == 0:
			print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))

	# test & one-hot encoding
	a = sess.run(hypothesis, feed_dict={X:[[1, 11, 7]]})
	print(a, sess.run(tf.arg_max(a, 1)))  # one hot encoding

	b = sess.run(hypothesis, feed_dict={X:[[1, 3, 4]]})
	print(b, sess.run(tf.arg_max(b, 1)))

	c = sess.run(hypothesis, feed_dict={X:[[1, 1, 0]]})
	print(c, sess.run(tf.arg_max(c, 1)))

	all = sess.run(hypothesis, feed_dict={X:[ [1, 11, 7], [1, 3, 4], [1, 1, 0] ]})
	print(all, sess.run(tf.arg_max(all, 1)))
