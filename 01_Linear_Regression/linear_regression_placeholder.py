import tensorflow as tf


x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

# try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 1 and b 0, butTensorflow will
# figure that out for us.)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# place holder
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis
hypothesis = W * X + b

# Simplified cost function
# get avg
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
a = tf.Variable(0.1) # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# Before starting, initialize the values. we will 'run' this first.
init = tf.initialize_all_variables()

# launch the graph
sess = tf.Session()
sess.run(init)

# fit the line
for step in range(2001):
	sess.run(train, feed_dict={X:x_data, Y:y_data})
	if step % 20 == 0:
		print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W), sess.run(b))

print(sess.run(hypothesis, feed_dict={X:5}) )
print(sess.run(hypothesis, feed_dict={X:2.5}) )
