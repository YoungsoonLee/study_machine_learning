import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


env = gym.make('CartPole-v0')

# input and output size based on the Env
# input_size = env.observation_space.n
input_size = env.observation_space.shape[0] # 4
output_size = env.action_space.n # 2
learning_rate = 1e-1

# these lines establish the feed-forward part of the network used to choose actions
X = tf.placeholder(shape=[None, input_size], dtype=tf.float32) # state input
#W = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.01)) # weight ~!!!
W = tf.get_variable('W1', shape=[input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())

Qpred = tf.matmul(X, W) # out Q predictions
Y = tf.placeholder(shape=[None, output_size], dtype=tf.float32) # Y label

#cost
loss = tf.reduce_sum(tf.square(Y - Qpred))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

dis = 0.99
num_episodes = 5000
rList = []

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(num_episodes):
	s = env.reset()
	e = 1. / ((i/10)+1)
	done = False
	#local_loss = []
	step_count = 0 

	# the Q-Network training
	while not done:
		step_count += 1
		x = np.reshape(s, [1, input_size])

		# choose an action by greedily
		Qs = sess.run(Qpred, feed_dict={X: x})
		#print(Qs)
		if np.random.rand(1) < e:
			a = env.action_space.sample()
		else:
			a = np.argmax(Qs)

		# get new state and reward from environment
		s1, reward, done, _ = env.step(a)
		if done:
			# update Q, and no Qs+1, since it's a terminal state
			Qs[0, a] = -100
		else:
			x1 = np.reshape(s1, [1, input_size])
			# Otain the Q_s1 values by feeding the new state through our network
			Qs1 = sess.run(Qpred, feed_dict={X: x1})
			# update Q
			Qs[0, a] = reward + dis * np.max(Qs1)

		# Train our network using target (Y) and predicted Q (Qpred) values
		sess.run(train, feed_dict={X: x, Y: Qs})

		#rAll += reward
		s = s1
	rList.append(step_count)
	print('Episode: {} steps: {}'.format(i, step_count))

	if len(rList) > 10 and np.mean(rList[-10:]) > 500:
		break

# See our trained network in action
observation = env.reset()
reward_sum = 0
while True:
	env.render()

	x = np.reshape(observation, [1, input_size])
	Qs = sess.run(Qpred, feed_dict={X: x})
	a = np.argmax(Qs)

	observation, reward, done, _ = env.step(a)
	reward_sum += reward
	if done:
		print('Total score: {}'.format(reward_sum))
		break
