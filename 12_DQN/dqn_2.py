import numpy as np
import tensorflow as tf
import random
import dqn
from collections import deque
import gym

env = gym.make('CartPole-v0')

# Constants defining our nenural network
input_size = env.observation_space.shape[0] # 4
output_size = env.action_space.n # 2

dis = 0.9
REPLAY_MEMORY = 50000


def replay_train(mainDQN, targetDQN, train_batch):
	x_stack = np.empty(0).reshape(0, mainDQN.input_size)
	y_stack = np.empty(0).reshape(0, mainDQN.output_size)

	# get stored information from the buffer
	for state, action, reward, next_state, done in train_batch:
		Q = mainDQN.predict(state)

		# terminal?
		if done:
			Q[0, action] = reward
		else:
			# obtain the Q` values by feeding the new state through our network
			Q[0, action] = reward + dis * np.max(targetDQN.predict(next_state))

		y_stack = np.vstack([y_stack, Q])
		x_stack = np.vstack([x_stack, state])

	# train our network using target and predicted Q values on each episode
	return mainDQN.update(x_stack, y_stack) # !!!!

# bot_play
def bot_play(mainDQN):
	# See our trained network in action
	s = env.reset()
	reward_sum = 0
	while True:
		env.render()
		a = np.argmax(mainDQN.predict(s))
		s, reward, done, _ = env.step(a)
		reward_sum += reward
		if done:
			print('Total score: {}'.format(reward_sum))
			break

def get_copy_var_ops(*, dest_scope_name='target',  src_scope_name='main'):
	# copy variables src_scope to dest_scope
	op_holder = []

	src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
	dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

	for src_var, dest_var in zip(src_vars, dest_vars):
		op_holder.append(dest_var.assign(src_var.value()))
	return op_holder



def main():
	max_episodes = 5000

	# store the previous observations in replay memory
	replay_buffer = deque()

	with tf.Session() as sess:
		mainDQN = dqn.DQN(sess, input_size, output_size, name='main')
		targetDQN = dqn.DQN(sess, input_size, output_size, name='target')
		tf.global_variables_initializer().run()

		# initialcopy q_net -> target_net
		copy_ops = get_copy_var_ops(dest_scope_name='target', src_scope_name='main')
		sess.run(copy_ops)

		for episode in range(max_episodes):
			e = 1. / ((episode / 10) + 1)
			done = False
			step_count = 0

			state = env.reset()

			while not done:
				if np.random.rand(1) < e:
					action = env.action_space.sample()
				else:
					# choose an action by greedily from the Q-network
					action = np.argmax(mainDQN.predict(state))

				# Get new state and reward from environment
				next_state, reward, done, _ = env.step(action)
				if done: # big penalty
					reward = -100

				# save the experience to our buffer
				replay_buffer.append((state, action, reward, next_state, done))
				if len(replay_buffer) > REPLAY_MEMORY:
					replay_buffer.popleft()

				state = next_state
				step_count += 1
				if step_count > 10000: # good enough
					break

			print('Episode: {} steps: {}'.format(episode, step_count))
			if step_count > 10000:
				pass

			if episode % 10 == 1: # train every 10 episodes
				# get a random batch of experiences.
				for _ in range(50):
					# minibatch works better
					minibatch = random.sample(replay_buffer, 10)
					loss, _ = replay_train(mainDQN, targetDQN, minibatch)
				print('loss: ', loss)
				# copy q_net -> target_net
				sess.run(copy_ops)

		bot_play(mainDQN)

if  __name__ == '__main__':
	main()

