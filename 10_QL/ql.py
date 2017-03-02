import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

def rargmax(vector):
	""" Argmax that chooses randomly among eligible maximum indeices."""
	m = np.amax(vector)
	indices = np.nonzero(vector == m)[0]
	return pr.choice(indices)

register(
	id='FrozenLake-v3',
	entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
	kwargs = {'map_name': '4x4', 'is_slippery': False}
	)

env = gym.make('FrozenLake-v3')

# initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n]) # 16*4
# discount factor
dis = 0.99
# set learning parameters
num_episodes = 2000 # loop learning count

# create lists to contain total rewards and steps per episode
rList = []
for i in range(num_episodes):
	# e = 1. / ((i//100)+1) # for integer python2&3
	# reset environment and get first new observation
	state = env.reset()
	rAll = 0
	done = False

	# the Q-Table learnig alogrithm
	while not done:
		# E-greedy
		#if np.random.rand(1) < e:
		#	action = env.action_space.sample()
		#else:
			# action = rargmax(Q[state, :])
		
		# noise
		action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i+1) ) 

		# Get new state and reward from environment
		new_state, reward, done, _ = env.step(action)

		# !!!
		# Update Q-Table with new knowledge using learning rate
		# Discount factor
		#Q[state, action] = reward + np.max(Q[new_state, :]) # action is all
		Q[state, action] = reward + dis * np.max(Q[new_state, :]) # action is all
		
		rAll += reward
		state = new_state

	rList.append(rAll)

print('Success rate: ' + str(sum(rList)/num_episodes))
print('Final Q-Table Values')
print('LEFT DOWN RIGHT UP')
print(Q)
plt.bar(range(len(rList)), rList, color='blue')
plt.show()
