# coding:utf-8

import random
import copy
import numpy as np


class WindyEnv:
	"""
	Attributes:
	-wind : np.array([y,x]) containing wind speed
	-goal : np.array([y,x]), indexes of the final point
	"""
	def __init__(self, wind,goal,start):
		"""
		inputs :
		wind : numpy array of wind values 
		goal : array of int, indexes of the goal on the grid
		start : array of int, indexes of the starting point of the agent
		"""
		self.wind = wind
		self.goal = goal
		self.ylimit, self.xlimit = wind.shape[0]-1, wind.shape[1]-1
		self.start = start

	def step(self,old_state,action):
		"""
		state (np.array)
		action (str)
		returns reward & new state
		"""
		#update the state with the action
		#action_list = ['left', 'right', 'down', 'up']
		#operation_list = [[0,-1],[0,1],[1,0],[-1,0]]
		#state += operation_list[action_list.index(action)]
		state = copy.deepcopy(old_state)
		if action == "left":
			state[1] -=1
		if action == "right":
			state[1] +=1
		if action == "down":
			state[0] +=1
		if action == "up":
			state[0] -=1

		#apply wind force
		if state[0] <= self.ylimit and state[1] <= self.xlimit:
			state[0] -= self.wind[tuple(state)]


		# limit x,y to the borders of the grid
		if state[1] < 0 :
			state[1] = 0

		elif state[1] > self.xlimit:
			state[1] = self.xlimit

		if state[0] < 0:
			state[0] = 0

		elif state[0] >self.ylimit:
			state[0] = self.ylimit

		#compute reward
		#reward = 10/(1+np.linalg.norm(state - self.goal))
		if state == self.goal:
			reward = 0
		else :
			reward = -1
			#reward = -( (state[0]-self.goal[0])**2 + (state[1] - self.goal[1])**2 ) 
		return reward, state

class agent():
	def __init__(self, epsilon,env, gamma, alpha,q=None):
		self.actions = ['left', 'right', 'down', 'up']
		self.epsilon = epsilon
		self.state = copy.deepcopy(env.start)
		self.gamma = gamma
		self.alpha = alpha
		if type(q) == type(None):
			#print("init q")
			shape = env.wind.shape + (4,)
			self.q = -np.ones(shape)
			self.q[tuple(env.goal)] = 0
		else:
			self.q = copy.deepcopy(q)

	def choose_action(self):
		"""
		choose the action
		randomly or the one the maximise q
		"""

		if random.random() < self.epsilon:
			action = random.choice(self.actions)

		else:
			reduced_q = self.q[tuple(self.state)]
			q_max = np.max(reduced_q)
			max_action_idx = np.where(reduced_q== q_max)[0]
			action = self.actions[random.choice(max_action_idx)]

		return action


	def show(self,env):
		"""
		show state array
		"""
		array = np.zeros(env.wind.shape)
		array[tuple(self.state)] =1
		print(array)
		#return array

class qlearning_agent(agent):
	"""
	class for qlearning agent
	heriting from agent class
	"""
	def __init__(self, epsilon,env, gamma, alpha,q=None):
		agent.__init__(self, epsilon,env, gamma, alpha,q=q)
		self.type = "qlearning"

	def fitstep(self, env):
		"""
		choose an action
		update the state
		update the q array
		"""
		action = self.choose_action()
		action_idx = self.actions.index(action)
		old_state = copy.deepcopy(self.state)

		reward, self.state = env.step(self.state, action)

		self.q[tuple(old_state) + (action_idx,)] += self.alpha * (reward + self.gamma * self.q[tuple(self.state)].max() - self.q[tuple(old_state)+ (action_idx,)])


class sarsa_agent(agent):
	"""
	class for sarsa agent
	heriting from agent class
	"""
	def __init__(self, epsilon,env, gamma, alpha,q=None):
		agent.__init__(self, epsilon,env, gamma, alpha,q=q)
		self.type = "sarsa"

		# make a first step to initialize previous action, reward and state
		self.previous_action = self.choose_action()
		self.previous_state = copy.deepcopy(self.state)
		self.previous_reward, self.state = env.step(self.state, self.previous_action)

	def fitstep(self, env):
		"""
		choose an action
		update the state
		update the q array
		"""
		# compute new action, and get indexes of previous & current actions
		action = self.choose_action()
		action_idx = self.actions.index(action)
		previous_action_idx = self.actions.index(self.previous_action)

		self.previous_state = copy.deepcopy(self.state)
		reward, self.state = env.step(self.state, action)

		#update q values
		self.q[tuple(self.previous_state) + (previous_action_idx,)] += self.alpha * (self.previous_reward + self.gamma * self.q[tuple(self.state)  + (action_idx,)] - self.q[tuple(self.previous_state)+ (previous_action_idx,)])

		# update preivous_* attributes
		self.previous_reward = reward
		self.previous_action = action


if __name__ == '__main__':

	# create wind array
	shape = (7,10)
	wind = np.zeros(shape,dtype=int)
	wind[:,[3,4,5,8]] = 1
	wind[:,[6,7]] = 2

	#goal location (y,x)
	goal = [3,7]

	#starting point (y,x)
	start = [3,0]

	#create env :
	env = WindyEnv(wind, goal, start)

	epsilon = 0.1
	gamma = 0.9
	alpha = 0.5
	q=None

	N_gen = 100
	max_step = 1000
	for n in range(N_gen):
		#instantiate a new agent with already learned q
		agent = qlearning_agent(epsilon, env,gamma,alpha,q=q)
		#agent = sarsa_agent(epsilon, env, gamma, alpha, q)
		step = 0
		while step<max_step and agent.state != env.goal:
			step+=1
			agent.fitstep(env)

		q = copy.deepcopy(agent.q)
		print(n,"gen.,\n", step,"step(s)")
		agent.show(env)

	print(q)