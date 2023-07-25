from model import Critic, Actor
import torch as th
from copy import deepcopy      
from memory import ReplayMemory, Experience
from torch.optim import Adam
from randomProcess import OrnsteinUhlenbeckProcess
import torch.nn as nn
import numpy as np
from params import scale_reward
from compare_algorithm import utils

def soft_update(target, source, t):                                      #t->软更新参数
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class JODRL_PP:
    def __init__(self, n_agents, dim_obs, dim_act, batch_size,
                 capacity, episodes_before_train):                      #capacity经验回放池容量，episode_before_train缓存池多少容量开始训练
        self.actors = [Actor(dim_obs, dim_act) for i in range(n_agents)]        #每个agent的obs和act组成的列表
        self.critics = [Critic(n_agents, dim_obs,
                               dim_act) for i in range(n_agents)]               #每个critic都包含n个agents
        self.actors_target = deepcopy(self.actors)                      #深复制，新的对象
        self.critics_target = deepcopy(self.critics)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.use_cuda = th.cuda.is_available()
        self.episodes_before_train = episodes_before_train
        self.noise = utils.OrnsteinUhlenbeckActionNoise(1)      
        self.GAMMA = 0.99
        self.tau = 0.01                                         

        self.var = [1 for i in range(n_agents)]

        # 创建优化器
        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=0.0001) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.0001) for x in self.actors]
        
        # 创建学习率调度器
        self.critic_scheduler = [th.optim.lr_scheduler.StepLR(x, step_size=100, gamma=0.8) for x in self.critic_optimizer]
        self.actor_scheduler = [th.optim.lr_scheduler.StepLR(x, step_size=100, gamma=0.8) for x in self.actor_optimizer]

        if self.use_cuda:                     #使用GPU
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

        self.steps_done = 0
        self.episode_done = 0

    def update_policy(self):
        # do not train until exploration is enough
        if self.episode_done <= self.episodes_before_train:
            return None, None

        ByteTensor = th.cuda.ByteTensor if self.use_cuda else th.ByteTensor    
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor

        c_loss = []
        a_loss = []
        for agent in range(self.n_agents):        #对于每一个agent
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))              #解包
            non_final_mask = ByteTensor(list(map(lambda s: s is not None,
                                                 batch.next_states)))      #map配合匿名函数，标记 batch 中所有下一个状态是否为最终状态
            # state_batch: batch_size x n_agents x dim_obs
            state_batch = th.stack(batch.states).type(FloatTensor)
            action_batch = th.stack(batch.actions).type(FloatTensor)
            reward_batch = th.stack(batch.rewards).type(FloatTensor)
            # : (batch_size_non_final) x n_agents x dim_obs
            non_final_next_states = th.stack(
                [s for s in batch.next_states
                 if s is not None]).type(FloatTensor)

            # for current agent
            whole_state = state_batch.view(self.batch_size, -1)         #维度重构，-1表示转化为一维
            whole_action = action_batch.view(self.batch_size, -1)

            #Critic
            self.critic_optimizer[agent].zero_grad()                    #对每一个单独的agent，维度清零
            current_Q = self.critics[agent](whole_state, whole_action)   #根据所有智能体的状态和动作评估Q值

            non_final_next_actions = [                                          #每个智能体的目标Actor网络来生成对应的下一个状态下的动作
                self.actors_target[i](non_final_next_states[:,
                                                            i,
                                                            :]) for i in range(
                                                                self.n_agents)]
            non_final_next_actions = th.stack(non_final_next_actions)
            non_final_next_actions = (
                non_final_next_actions.transpose(0,                     #交换维度
                                                 1).contiguous())

            target_Q = th.zeros(
                self.batch_size).type(FloatTensor)

            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_states.view(-1, self.n_agents * self.n_states),
                non_final_next_actions.view(-1,
                                            self.n_agents * self.n_actions)
            ).squeeze()
            # scale_reward: to scale reward in Q functions

            target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (
                reward_batch[:, agent].unsqueeze(1) * scale_reward)     

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()
            self.critic_optimizer[agent].step()

            ##Actor
            self.actor_optimizer[agent].zero_grad()
            state_i = state_batch[:, agent, :]
            action_i = self.actors[agent](state_i)              #算出当前状态下的动作
            ac = action_batch.clone()
            ac[:, agent, :] = action_i
            whole_action = ac.view(self.batch_size, -1)         #把这个agent的动作换了一下
            actor_loss = -self.critics[agent](whole_state, whole_action)    #Critic网络评估值
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            self.actor_optimizer[agent].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        if self.steps_done % 100 == 0 and self.steps_done > 0:       #每一百步更新一下target net
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss
    
    #选择动作
    def select_action(self, state_batch, episode):
        # state_batch: n_agents x state_dim
        actions = th.zeros(
            self.n_agents,
            self.n_actions)
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        for i in range(self.n_agents): # 循环每个agent
            sb = state_batch[i, :].detach() # 每个agent的单独状态
            act = self.actors[i](sb.unsqueeze(0)).squeeze() # actor输出的动作

            #动作的探索性
            # act += th.from_numpy(  # 动作加上生成的噪音
            #     np.random.randn(self.n_actions) * self.var[i]).type(FloatTensor)
            act += th.from_numpy(  # 动作加上生成的噪音
                self.noise.sample(episode)).type(FloatTensor)
            if self.episode_done > self.episodes_before_train and self.var[i] > 0.05:       #开始训练了
                self.var[i] *= 0.99999

            # act = th.clamp(act, -1.0, 1.0) 
            act = th.clamp(act, 0, 2.0) # 动作裁剪到0，2


            actions[i, :] = act  # 更新action
        self.steps_done += 1

        return actions
