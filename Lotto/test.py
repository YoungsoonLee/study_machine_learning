import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[1:-1]
y_data = xy[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([1], -50.0, 50.0))  # need to smae form with x ~! 
W2 = tf.Variable(tf.random_uniform([1], -50.0, 50.0))  # need to smae form with x ~! 
W3 = tf.Variable(tf.random_uniform([1], -50.0, 50.0))  # need to smae form with x ~! 
W4 = tf.Variable(tf.random_uniform([1], -50.0, 50.0))  # need to smae form with x ~! 
W5 = tf.Variable(tf.random_uniform([1], -50.0, 50.0))  # need to smae form with x ~! 
W6 = tf.Variable(tf.random_uniform([1], -50.0, 50.0))  # need to smae form with x ~! 
W7 = tf.Variable(tf.random_uniform([1], -50.0, 50.0))  # need to smae form with x ~! 

# our hypothesis
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
print(x_data[0])
# our hypothesis
hypothesis = W1*x_data[0] + W2*x_data[1] + W3*x_data[2] + W4*x_data[3] + W5*x_data[4] + W6*x_data[5] + W7*x_data[6]  + b # muliple matrix !

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# minimize
a = tf.Variable(0.1)  # learning rate, alpha. step size
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

# launch the graph
with tf.Session() as sess:
	sess.run(init)

	for step in range(100):
		sess.run(train, feed_dict={X:x_data, Y:y_data})
		#if step % 200 == 0:
		print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W1), sess.run(W2), sess.run(W3), sess.run(W4), sess.run(W5), sess.run(W6) )

	# test & one-hot encoding
	# a = sess.run(hypothesis, feed_dict={X:[[1, 11, 7, 8, 21, 34, 42, 46]]})
	# print(a, sess.run(tf.arg_max(a, 1)))
	a = sess.run(hypothesis, feed_dict={X:[ [1, 11, 7, 8, 21, 34, 42, 46] ]}) > 0.5
	print(a[0])
	print( a.all())

