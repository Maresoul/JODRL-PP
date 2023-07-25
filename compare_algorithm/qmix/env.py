from gym.core import Env
import argparse
import numpy as np


def qmix_args(args):
    args.rnn_hidden_dim = 64
    args.two_hyper_layers = False
    args.qmix_hidden_dim = 32
    args.lr = 0.0001

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.05
    anneal_steps = 50000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the epoch to train the agent
    args.n_epoch = 10

    # the number of the episodes in one epoch
    args.n_episodes = 100

    # the number of the train steps in one epoch
    args.train_steps = 1

    # # how often to evaluate
    args.evaluate_cycle = 100

    # experience replay
    args.batch_size = 64
    args.buffer_size = int(5e3)

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 20

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


def get_common_args():
    parser = argparse.ArgumentParser()

    # the environment setting
    parser.add_argument('--obs_space', type=int, default=3, help='observation space')
    parser.add_argument('--state_space', type=int, default=30, help='observation space')

    parser.add_argument('--action_space', type=int, default=10, help='action space')
    parser.add_argument('--num_actions', type=int, default=10, help='number of agents')
    parser.add_argument('--num_agents', type=int, default=10, help='number of agents')
    parser.add_argument('--max_episode_steps', type=int, default=300, help='number of agents')

    parser.add_argument('--difficulty', type=str, default='7', help='the difficulty of the game')
    parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
    parser.add_argument('--map', type=str, default='3m', help='the map of the game')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='', help='the directory of save the replay')

    parser.add_argument('--alg', type=str, default='qmix', help='the algorithm to train the agent')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=False, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="Adam", help='optimizer')
    parser.add_argument('--n_evaluate_episode', type=int, default=3, help='number of the episode to evaluate the agent')
    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--learn', type=bool, default=True, help='whether to train the model')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')
    parser.add_argument('--threshold', type=float, default=19.9, help='threshold to judge whether win')
    parser.add_argument('--cem_sample_N', type=int, default=64, help='cem_sample_arg')
    parser.add_argument('--cem_sample_Ne', type=int, default=6, help='cem_sample_arg')
    parser.add_argument('--max_ratio', type=float, default=1.1, help='smooth the reward curve')
    parser.add_argument('--smooth_ratio', type=float, default=1.05, help='smooth the reward curve')
    args = parser.parse_args()
    return args


# class RandomEnv(Env):

#     def __init__(self, args):
#         """
#         :param args:
#             args.obs_space , int 10
#             args.action_space , int, 3
#             args.num_agents, int , 4
#         """
#         super(RandomEnv, self).__init__()
#         self.args = args
#         self.action_space = args.action_space
#         self.obs_space = args.obs_space
#         self.state_space = args.state_space
#         self.num_agent = args.num_agents

#         self.observation = None
#         self.state = None

#         self.reset()

#     def reset(self):
#         self.max_episode_steps = np.random.randint(0, self.args.max_episode_steps, 1)[0]
#         self.current_step = 0
#         self.done = False

#         self.observation = np.random.randn(self.num_agent, self.obs_space)
#         self.state = np.random.randn(self.state_space)

#         return self.observation, self.state

#     def step(self, actions):
#         assert len(actions) == self.num_agent

#         if self.current_step >= self.max_episode_steps:
#             self.done = True
#             #print('current step : {}'.format(self.current_step))

#         self.current_step = self.current_step + 1

#         self.observation = np.random.randn(self.num_agent, self.obs_space)
#         self.state = np.random.randn(self.state_space)

#         # original - self.state, self.observation, np.random.randn(self.num_agent), self.done, []
#         # local reward - np.random.randn(self.num_agent), self.done, []
#         return np.random.randn(), self.done, []

#     def get_obs(self):
#         return self.observation

#     def get_state(self):
#         return self.state