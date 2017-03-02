import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def one_hot(x):
	return np.identity(16)[x:x+1]

env = gym.make('FrozenLake-v0')

# input and output size based on the Env
input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1

# these lines establish the feed-forward part of the network used to choose actions
X = tf.placeholder(shape=[1, input_size], dtype=tf.float32) # state input
W = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.01)) # weight ~!!!

Qpred = tf.matmul(X, W) # out Q predictions
Y = tf.placeholder(shape=[1, output_size], dtype=tf.float32) # Y label

#cost
loss = tf.reduce_sum(tf.square(Y - Qpred))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

dis = 0.99
num_episodes = 2000

rList = []

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for i in range(num_episodes):
		s = env.reset()
		e = 1. / ((i/50)+10)
		rAll = 0
		done = False
		local_loss = []

		# the Q-Network training
		while not done:
			# choose an action by greedily
			Qs = sess.run(Qpred, feed_dict={X: one_hot(s)})
			#print(Qs)
			if np.random.rand(1) < e:
				a = env.action_space.sample()
			else:
				a = np.argmax(Qs)

			# get new state and reward from environment
			s1, reward, done, _ = env.step(a)
			if done:
				# update Q, and no Qs+1, since it's a terminal state
				Qs[0, a] = reward
			else:
				# Otain the Q_s1 values by feeding the new state through our network
				Qs1 = sess.run(Qpred, feed_dict={X: one_hot(s1)})
				# update Q
				Qs[0, a] = reward + dis * np.max(Qs1)

			# Train our network using target (Y) and predicted Q (Qpred) values
			sess.run(train, feed_dict={X: one_hot(s), Y: Qs})

			rAll += reward
			s = s1
		rList.append(rAll)

print('Success rate: ' + str(sum(rList)/num_episodes) + '%')
plt.bar(range(len(rList)), rList, color='blue')
plt.show()

