import tensorflow as tf


x1_data = [[0., 2., 0., 4., 0.],[1., 0., 3., 0., 5.]] # matrix !
y_data = [1, 2, 3, 4, 5]

# try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 1 and b 0 , but TF wull
# figure that out for us.)

W = tf.Variable(tf.random_uniform([1,2], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# our hypothesis
hypothesis = tf.matmul(W, x1_data) + b # muliple matrix !

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# Minimize
a = tf.Variable(0.1) # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# Before staring, initialize the variables, we will 'run' this first.
init = tf.initialize_all_variables()

# launch the graph
sess = tf.Session()
sess.run(init)

# fit the lune.
for step in range(2001):
	sess.run(train)
	if step % 20 == 0:
		print(step, sess.run(cost), sess.run(W), sess.run(b))