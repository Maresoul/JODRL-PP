from __future__ import division
from re import A
import numpy as np
import torch
from torch.autograd import Variable
import gc
from MEC_env import mec_env
import train
import buffer

import os

def getnear(state):
	server = np.array([[333,333], [333,666], [666,333], [666,666]])
	distance = 10000
	near = 1
	for i in range(4):
		dis = pow((pow(state[1] - server[i][0], 2) + pow(state[2]- server[i][1],2)), 0.5)
		if dis < distance :
			distance = dis
			near = i
	return near+1

# 使用GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# env = gym.make('Pendulum-v0')
reward_record = []
delay_record = []
energy_record = []
privacy_record = []
punish_record = []
MAX_EPISODES = 300
MAX_STEPS = 200
MAX_BUFFER = 1000000

n_agents = 50
n_actions = 10
n_states = 3
S_DIM = n_agents * n_states
A_DIM = n_agents * n_actions
A_MAX = 1


env = mec_env(n_agents, n_states , n_actions, task_rate = 2)


for _ep in range(MAX_EPISODES):
	observation = env.reset()
	total_reward = 0.0
	total_privacy = 0.0
	total_energy = 0.0
	total_delay = 0.0
	total_punish = 0.0
	for r in range(MAX_STEPS):
		state = np.float32(observation)
		action = []

		for i in range(n_agents):
			near = getnear(state[i])
			temp = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])
			temp[near] = 1
			temp[near + 5] = 1
			action.append(temp)
		new_observation, reward, done, _info, action = env.step(action)

		# # dont update if this is validation
		# if _ep%50 == 0 or _ep>450:
		# 	continue
		total_reward += sum(reward)
		total_privacy += sum(_info[0])
		total_energy += sum(_info[1])
		total_delay += sum(_info[2])
		total_punish += sum(_info[3])

		observation = new_observation

	temp = n_agents * MAX_STEPS
	print('Episode: %d, reward = %f' % (_ep, total_reward/temp))
	reward_record.append(total_reward/temp)
	privacy_record.append(total_privacy/temp)
	energy_record.append(total_energy/temp)
	delay_record.append(total_delay/temp)
	punish_record.append(total_punish/temp)
	# check memory consumption and clear memory
	gc.collect()
	# process = psutil.Process(os.getpid())
	# print(process.memory_info().rss)

	# if _ep%100 == 0:
	# 	trainer.save_models(_ep)

np.save('near_reward', reward_record)
np.save('near_privacy', privacy_record)
np.save('near_energy', energy_record)
np.save('near_delay', delay_record)
np.save('near_punish', punish_record)