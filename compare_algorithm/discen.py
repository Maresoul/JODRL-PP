from __future__ import division
from re import A
import numpy as np
import torch
from torch.autograd import Variable
import gc
from MEC_env import mec_env
import train
import buffer
import time

import os



# 使用GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# env = gym.make('Pendulum-v0')
reward_record = []
delay_record = []
energy_record = []
privacy_record = []
punish_record = []
time_record = []
# MAX_EPISODES
# MAX_STEPS 
MAX_EPISODES = 500
MAX_STEPS = 300
MAX_BUFFER = 1000000

n_agents = 100
n_actions = 10
n_states = 3
S_DIM = n_states
A_DIM = n_actions
A_MAX = 1


env = mec_env(n_agents, n_states , n_actions, task_rate = 2)
ram = []
trainer = []
for i in range(n_agents):
    ram.append(buffer.MemoryBuffer(MAX_BUFFER))
    trainer.append(train.Trainer(S_DIM, A_DIM, A_MAX, ram[i]))
start_time = time.time()
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
			action.append(trainer[i].get_exploration_action(state[i], _ep)) #组装action
		# if _ep%5 == 0:
		# 	# validate every 5th episode
		# 	action = trainer.get_exploitation_action(state)
		# else:
		# 	# get action based on observation, use exploration policy here
		# 	action = trainer.get_exploration_action(state)
		# n X action

		new_observation, reward, done, _info, new_action = env.step(action)
		# # dont update if this is validation
		# if _ep%50 == 0 or _ep>450:
		# 	continue
		total_reward += sum(reward)
		total_privacy += sum(_info[0])
		total_energy += sum(_info[1])
		total_delay += sum(_info[2])
		total_punish += sum(_info[3])
		new_state = np.float32(new_observation)
			# push this exp in ram
		for i in range(n_agents):
			ram[i].add(state[i], new_action[i], reward[i], new_state[i])

		observation = new_observation
		# perform optimization
		if _ep >= 1 :
			for i in range(n_agents):
				trainer[i].optimize()

	end_time = time.time()
 
	for i in range(n_agents):				#衰减学习率
		trainer[i].scheduler_a.step()	
		trainer[i].scheduler_c.step()	
	print("Current learning rate:", trainer[0].scheduler_a.get_last_lr()[0])

	temp = n_agents * MAX_STEPS
	print('Episode: %d, reward = %f' % (_ep, total_reward/temp))
	reward_record.append(total_reward/temp)
	privacy_record.append(total_privacy/temp)
	energy_record.append(total_energy/temp)
	delay_record.append(total_delay/temp)
	punish_record.append(total_punish/temp)
	print(end_time-start_time)
	time_record.append(end_time-start_time)
	# check memory consumption and clear memory
	gc.collect()
	# process = psutil.Process(os.getpid())
	# print(process.memory_info().rss)

	# if _ep%100 == 0:
	# 	trainer.save_models(_ep)

print('Completed episodes')


np.save('Discen_privacy', privacy_record)
np.save('Discen_energy', energy_record)
np.save('Discen_delay', delay_record)
np.save('Discen_punish', punish_record)