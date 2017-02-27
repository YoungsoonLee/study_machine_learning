import tensorflow as tf
import numpy as np
import random

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([4742+5, 8], -1.0, 1.0))  # need to smae form with x ~! 

# sigmoid
#h = tf.matmul(W, X)
#hypothesis = tf.div(1., 1.+tf.exp(-h))  # sigmoid
#cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

# Simplified
hypothesis = tf.matmul(W, X)  # muliple matrix Simple~!!!
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# minimize
a = tf.Variable(0.1)  # learning rate, alpha. step size
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(2001):
	sess.run(train, feed_dict={X:x_data, Y:y_data})
	if step % 20 == 0:
		print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))


print('-'*50)

# get the number ~!!!!
go = True
result =[]
step = 0
base_number = [x for x in range(1,50)]
while go:
	rnd = random.sample(base_number, 7)
	i_data = [1]
	i_data += rnd
	i_data = np.array(i_data)

	a = sess.run(hypothesis, feed_dict={X: i_data.reshape(8, 1) }) > 0.5
	if a.all():
		result = i_data.copy()
		go = False # break

	print(step, i_data, a)
	step += 1

print('-'*50)
print(result)


