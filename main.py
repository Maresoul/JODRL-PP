from cmath import inf
from distutils.log import info
from compare_algorithm.MEC_env import mec_env
from JODRL_PP import JODRL_PP
import numpy as np
import torch as th
# import visdom
from params import scale_reward
import os
import time

# do not render the scene
e_render = False


# 使用GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
reward_record = []
delay_record = []
energy_record = []
privacy_record = []
punish_record = []
# np.random.seed(1234)
# th.manual_seed(1234)

n_agents = 10
n_states = 3
n_actions = 10              #2*4+2
capacity = 10000000
batch_size = 64

# n_episode 
# max_steps

n_episode = 1000
max_steps = 300
episodes_before_train = 1

win = None
param = None

world = mec_env(n_agents, n_states , n_actions, task_rate = 2)
#设置maddpg参数：agent个数（用户个数），状态维度，动作维度，batchsize，经验池容量，训练前需要累积的多少轮样本
jodrl_pp = JODRL_PP(n_agents, n_states, n_actions, batch_size, capacity,
                episodes_before_train)

FloatTensor = th.cuda.FloatTensor if jodrl_pp.use_cuda else th.FloatTensor
for i_episode in range(n_episode):
    obs = world.reset() #获取环境初始化状态
    obs = np.stack(obs) 
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()
    total_reward = 0.0
    total_privacy = 0.0
    total_energy = 0.0
    total_delay = 0.0
    total_punish = 0.0

    for t in range(max_steps):
        # # render every 100 episodes to speed up training
        # if i_episode % 100 == 0 and e_render:
        #     world.render()
        obs = obs.type(FloatTensor)
        action = jodrl_pp.select_action(obs, i_episode).data.cpu()

        obs_, reward, done, _info, action = world.step(action.numpy())
        action = th.from_numpy(action)
        reward_ = th.FloatTensor(reward).type(FloatTensor)
        obs_ = np.stack(obs_)
        obs_ = th.from_numpy(obs_).float()
        if t != max_steps - 1:
            next_obs = obs_
        else:
            next_obs = None

        total_reward += sum(reward)
        total_privacy += sum(_info[0])
        total_energy += sum(_info[1])
        total_delay += sum(_info[2])
        total_punish += sum(_info[3])

        jodrl_pp.memory.push(obs.data, action, next_obs, reward_)

        obs = next_obs
        c_loss, a_loss = jodrl_pp.update_policy()

    for scheduler in jodrl_pp.critic_scheduler:
        scheduler.step()
    for scheduler in jodrl_pp.actor_scheduler:
        scheduler.step()
    print("Current learning rate:", jodrl_pp.actor_scheduler[0].get_last_lr()[0])

    jodrl_pp.episode_done += 1
    temp = max_steps * n_agents
    print('Episode: %d, reward = %f' % (i_episode, total_reward/temp))
    
    
    reward_record.append(total_reward/temp)
    privacy_record.append(total_privacy/temp)
    energy_record.append(total_energy/temp)
    delay_record.append(total_delay/temp)
    punish_record.append(total_punish/temp)

    if jodrl_pp.episode_done == jodrl_pp.episodes_before_train:
        print('training now begins...')

np.save('JODRL_reward', reward_record)
np.save('JODRL_privacy', privacy_record)
np.save('JODRL_energy', energy_record)
np.save('JODRL_delay', delay_record)
np.save('JODRL_punish', punish_record)


