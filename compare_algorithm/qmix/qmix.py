import torch
import os
from qmix_net import RNN
from qmix_net import QMixNet


class QMIX:
    def __init__(self, args):
        self.num_actions = args.num_actions
        self.num_agents = args.num_agents
        self.state_space = args.state_space
        self.obs_space = args.obs_space
        input_shape = self.obs_space
        if args.last_action:
            input_shape += self.num_actions
        if args.reuse_network:
            input_shape += self.num_agents

        self.eval_rnn = RNN(input_shape, args)
        self.target_rnn = RNN(input_shape, args)
        self.eval_qmix_net = QMixNet(args)
        self.target_qmix_net = QMixNet(args)
        self.args = args
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_qmix_net.cuda()
            self.target_qmix_net.cuda()
        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map

        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/rnn_net_params.pkl'
                path_qmix = self.model_dir + '/qmix_net_params.pkl'
                self.eval_rnn.load_state_dict(torch.load(path_rnn))
                self.eval_qmix_net.load_state_dict(torch.load(path_qmix))
                print('Successfully load the model: {} and {}'.format(path_rnn, path_qmix))
            else:
                raise Exception("No model!")

        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_rnn.parameters())
        if args.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.eval_parameters, lr=args.lr)

        self.eval_hidden = None
        self.target_hidden = None
        print('Init QMIX')

    def learn(self, batch, max_episode_len, train_step, epsilon=None):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        batch_num = batch['o'].shape[0]  
        self.init_hidden(batch_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'], batch['u'], \
                                                             batch['r'],  batch['avail_u'], batch['avail_u_next'],\
                                                             batch['terminated']
        mask = 1 - batch["padded"].float()
        
        #这里reward存储为(batch_num,step,num_agent,r)，转变为(batch_num,step,r_sum)
        r = torch.sum(r, dim=2)

        
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)
        if self.args.cuda:
            s = s.cuda()
            u = u.cuda()
            r = r.cuda()
            s_next = s_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
            
        # q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
        # q_targets[avail_u_next == 0.0] = - 9999999
        # q_targets = q_targets.max(dim=3)[0]

        q_total_eval = self.eval_qmix_net(q_evals, s)
        q_total_target = self.target_qmix_net(q_targets, s_next)

        # targets = r + self.args.gamma * q_total_target * (1 - terminated)
        targets = r + self.args.gamma * q_total_target


        td_error = (q_total_eval - targets.detach())
        # masked_td_error = mask * td_error  # 抹掉填充的经验的td_error

        # loss = masked_td_error.pow(2).mean()
        # loss = (masked_td_error ** 2).sum() / mask.sum()
        loss = td_error.pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

    def _get_inputs(self, batch, transition_idx):
        obs, obs_next, u= batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u'][:, transition_idx]    
                        
        batch_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)

        if self.args.last_action:
            if transition_idx == 0:  
                inputs.append(torch.zeros_like(u))
            else:
                inputs.append(u)
            inputs_next.append(u)
        # if self.args.reuse_network:
        #     inputs.append(torch.eye(self.args.num_agents).unsqueeze(0).expand(episode_num, -1, -1)  )
        #     inputs_next.append(torch.eye(self.args.num_agents).unsqueeze(0).expand(episode_num, -1, -1))
        
        inputs = torch.cat([x.reshape(batch_num*self.args.num_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(batch_num*self.args.num_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next        

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)           #inputs,inputs_next-输入obs
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()
            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

            q_eval = q_eval.view(episode_num, self.num_agents, -1)
            q_target = q_target.view(episode_num, self.num_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)      
        return q_evals, q_targets           #(1,300,10,10)

    def init_hidden(self, batch_num):
        self.eval_hidden = torch.zeros((batch_num, self.num_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((batch_num, self.num_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_qmix_net.state_dict(), self.model_dir + '/' + num + '_qmix_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')
