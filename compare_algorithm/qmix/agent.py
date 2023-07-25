
import numpy as np
import torch
from qmix import QMIX
from torch.distributions import Categorical
import torch.distributions as tdist

    

class Agents:
    def __init__(self, args):
        self.num_actions = args.num_actions
        self.num_agents = args.num_agents
        self.state_space = args.state_space
        self.obs_space = args.obs_space
        self.policy = QMIX(args)
        self.args = args
        if self.args.cuda:
            self.ftype = torch.cuda.FloatTensor
        else:
            self.ftype = torch.FloatTensor  
        self.mu = self.ftype(1, self.num_agents, self.num_actions).zero_()   
        self.std = self.ftype(1, self.num_agents, self.num_actions).zero_() + 1.0  
        self.test_num = 0

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, evaluate=False):
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]

        agent_id = np.zeros(self.num_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))

        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))

        hidden_state = self.policy.eval_hidden[:, agent_num, :]
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        # avail_actions =torch. tensor(avail_actions, dtype=torch.float32).unsqueeze(0)

        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)

        # q_value[avail_actions == 0.0] = -float("inf")
        if np.random.uniform() < epsilon:
            # action = np.random.choice(avail_actions_ind)
            action = np.random.choice(self.args.num_actions)
        else:
            action = torch.argmax(q_value)

        return action
    
    
    def cem_sampling(self, obs, last_action, step, epoch):
        N = self.args.cem_sample_N  # Number of samples from the param distribution
        Ne = self.args.cem_sample_Ne  # Number of best samples we will consider
        maxits = 3 # NUmber for loop
        # inputs = obs.clone()
        obs = obs.clone().view(1,self.num_agents,self.obs_space)           #(1,10,3)
        last_action = last_action.view(1,self.num_agents,self.num_actions)
        inputs = np.concatenate((obs, last_action), axis=2)
        batch_size = inputs.shape[0]
        inputs = torch.tensor(np.array(inputs), dtype=torch.float32)
        inputs = inputs.unsqueeze(0).expand(N, *inputs.shape).contiguous().view(-1, inputs.shape[-1])
        
        if self.args.cuda:
            inputs = inputs.cuda()
            ftype = torch.cuda.FloatTensor
        else:
            ftype = torch.FloatTensor        

        
        its = 0
        
        while its < maxits:
            # 标准差不能为负数
            if epoch < int(self.args.n_epoch*0.4):    
                if step < int(self.args.max_episode_steps*0.8):
                    self.std = torch.sigmoid(self.std)
                else:
                    #控制均值，缩小探索空间
                    self.mu = torch.sigmoid(self.mu)        
                    self.std = torch.sigmoid(self.std)
            else:
                self.mu[self.mu <= 0] = 1e-6    
                self.std[self.std <= 0] = 1e-6
            dist = tdist.Normal(self.mu.view(-1, self.num_actions), self.std.view(-1, self.num_actions))
            actions = dist.sample((N,)).detach() # (N,batch_size*n_agents,n_actions)
            actions_prime = torch.sigmoid(actions)
            binarized = torch.where(actions_prime[:, :, -1] > 0.5, torch.ones_like(actions_prime[:, :, -1]), torch.zeros_like(actions_prime[:, :, -1]))
            actions_prime[:,:,-1] = binarized
            if its == 0 and step == 0:
                hidden_state = self.policy.eval_hidden.repeat(N, 1, 1)
            else:
                hidden_state = self.policy.eval_hidden
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()
            ret,hidden = self.policy.eval_rnn.forward(inputs, hidden_state)
            self.policy.eval_hidden = hidden
            out = ret.view(N, -1, 1)            #(64,10,1)
            topk, topk_idxs = torch.topk(out, Ne, dim=0)        #取前Ne个总q值最大的
            topk_idxs = topk_idxs.view(Ne,self.num_agents,1)
            #print(topk_idxs.shape,topk_idxs.repeat(1, 1, self.args.num_actions).shape,actions.shape)
            #print(topk_idxs.long().shape)
            self.mu = torch.mean(actions.gather(0, topk_idxs.repeat(1, 1, self.num_actions).long()), dim=0)
            self.std = torch.std(actions.gather(0, topk_idxs.repeat(1, 1, self.num_actions).long()), dim=0)
            # print(mu,std)
            its += 1

        topk, topk_idxs = torch.topk(out, 1, dim=0)
        topk_idxs = topk_idxs.view(1,self.num_agents,1)
        action_prime = torch.mean(actions.gather(0, topk_idxs.repeat(1, 1, self.num_actions).long()), dim=0)
        #action_prime = torch.mean(actions_prime.gather(0, topk_idxs.repeat(1, 1, self.args.num_actions).long()), dim=0)
        chosen_actions = action_prime.clone().view(batch_size, self.num_agents, self.num_actions).detach()

        return chosen_actions
    

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[1]

        max_episode_len = self.args.max_episode_steps

        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.max_episode_steps):
                if transition_idx + 1 >= max_episode_len:
                    max_episode_len = transition_idx + 1
                break

        return max_episode_len

    def train(self, batch, train_step, epsilon=None):
        max_episdoe_len = self._get_max_episode_len(batch)      #300

        for key in batch.keys():
            batch[key] = batch[key][:,:max_episdoe_len]

        self.policy.learn(batch, max_episdoe_len, train_step, epsilon)

        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)


