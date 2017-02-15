import tensorflow as tf
import numpy as np


xy = np.loadtxt('07train.txt', unpack=True)
# x_data = xy[0:-1]
# y_data = xy[-1]
x_data = np.transpose(xy[:-1])
y_data = np.reshape(xy[-1], (4, 1))

X = tf.placeholder(tf.float32, name='X-input')
Y = tf.placeholder(tf.float32, name='Y-input')

""" for test xor accuracy 50%
# weight
W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

# Our hypothesis
h = tf.matmul(W, X)
hypothesis = tf.div(1., 1.+tf.exp(-h))	# sigmoid
"""

""" for NN """
W1 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0), name='Weight1')
W2 = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0), name='Weight1')

b1 = tf.Variable(tf.zeros([2]), name="Bias1")
b2 = tf.Variable(tf.zeros([1]), name="Bias2")

# Our hypothesis
with tf.name_scope('layer2') as scope:
	L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
with tf.name_scope('layer3') as scope:
	hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)
""" end for NN """

# Cost function
with tf.name_scope('cost') as scope:
	cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
	cost_summ = tf.scalar_summary('cost', cost)

# Minimize
a = tf.Variable(0.1)  # learning rate, alpha
with tf.name_scope('train') as scope:
	optimizer = tf.train.GradientDescentOptimizer(a)
	train = optimizer.minimize(cost)

# add histogran
w1_hist = tf.histogram_summary('weight1', W1)
w2_hist = tf.histogram_summary('weight2', W2)

b1_hist = tf.histogram_summary('biases1', b1)
b2_hist = tf.histogram_summary('biases2', b2)

y_hist = tf.histogram_summary('y', Y)

# Before staring, initialize the variables. We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
	# tensorboard --logdir=./logs/xor_logs
	merged = tf.merge_all_summaries()
	writer = tf.train.SummaryWriter('./logs/xor_logs', sess.graph_def)

	sess.run(init)
	# fit the line
	for step in range(200000):
		sess.run(train, feed_dict={X:x_data, Y:y_data})
		if step % 2000 == 0:
			# b1과 b2는 출력 생략. 한 줄에 출력하기 위해 reshape 사용
			# r1, (r2, r3) = sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run([W1, W2])
			# print('{:5} {:10.8f} {} {}'.format(step+1, r1, np.reshape(r2, (1,4)), np.reshape(r3, (1,2))))
			# print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run([W1, W2]))
			summary = sess.run(merged, feed_dict={X:x_data, Y:y_data})
			writer.add_summary(summary, step)
		
	print('-'*50)
	# test model
	correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)
	# calculate accurary
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

	print(sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data}))
	print("Accuracy:", accuracy.eval({X:x_data, Y:y_data}))
