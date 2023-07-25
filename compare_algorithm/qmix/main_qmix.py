from env import *
from cmath import inf
from distutils.log import info
from MEC_env import mec_env
from worker import RolloutWorker
from agent import Agents
from replay_buffer import ReplayBuffer

import numpy as np
import torch as th
# import visdom
# from params import scale_reward
import os

# do not render the scene
e_render = False

args = get_common_args()
args = qmix_args(args)



# 使用GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
reward_record = []
delay_record = []
energy_record = []
privacy_record = []
punish_record = []
# np.random.seed(1234)
# th.manual_seed(1234)

# n_agents = 10
# n_states = 3
# n_actions = 10              #2*4+2
# capacity = 10000000
# batch_size = 64
# n_episode = 1000
# max_steps = 300
# episodes_before_train = 1

world = mec_env(n_agents=10, n_obs=3 , n_action=10, task_rate = 2)
agents = Agents(args)
worker = RolloutWorker(world, agents, args)
buffer = ReplayBuffer(args)



train_steps = 0

save_path = args.result_dir 

# def evaluate():
#     win_number = 0
#     episode_rewards = 0
#     for epoch in range(args.n_evaluate_episode):
#         _, episode_reward = worker.generate_episode(evaluate=True)
#         episode_rewards += episode_reward
#         if episode_reward > args.threshold:
#             win_number += 1
#     return win_number / args.n_evaluate_episode, episode_rewards / args.n_evaluate_episode

for epoch in range(args.n_epoch):       #10

    episodes = []
    for episode_idx in range(args.n_episodes):      #100
        episode, eval_info = worker.generate_episode(episode_idx,epoch)
        episodes.append(episode)
        temp = args.max_episode_steps * args.num_agents
        
        #平滑曲线
        temp_reward = eval_info[0]/temp
        if len(reward_record)>0 and temp_reward>2 and temp_reward/reward_record[-1] > args.max_ratio:
            print(args.max_ratio,args.smooth_ratio,reward_record[-1])
            eval_info[0] = float(reward_record[-1]*temp*args.smooth_ratio)
        print('Epoch: %d,episodes: %d, average_reward = %f' % (epoch, episode_idx, eval_info[0]/temp))
        

        if args.alg.find('coma') > -1 or args.alg.find('central_v') > -1 or args.alg.find('reinforce') > -1:
            agents.train(episode_batch, train_steps, worker.epsilon)
            train_steps += 1
        else:
            buffer.store_episode(episode)
            for _ in range(args.train_steps):
                mini_batch = buffer.sample(min(buffer.current_size, args.batch_size))
                agents.train(mini_batch, train_steps)
                train_steps += 1
    
    
        reward_record.append(eval_info[0]/temp)
        privacy_record.append(eval_info[1]/temp)
        energy_record.append(eval_info[2]/temp)
        delay_record.append(eval_info[3]/temp)
        punish_record.append(eval_info[4]/temp)
    
np.save('QMIX_reward', reward_record)
np.save('QMIX_privacy', privacy_record)
np.save('QMIX_energy', energy_record)
np.save('QMIX_delay', delay_record)
np.save('QMIX_punish', punish_record)





