import gym
from gym import spaces
import numpy as np
from envs.env_core import EnvCore


class ContinuousActionEnv(object):
    """
    对于连续动作环境的封装
    Wrapper for continuous action environment.
    """

    def __init__(self,all_args):
        self.env = EnvCore(all_args)
        self.num_agent = self.env.agent_num

        self.signal_obs_dim = self.env.obs_dim
        self.signal_action_dim = self.env.action_dim

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False

        self.movable = True

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        share_obs_dim = 0
        total_action_space = []
        for agent in range(self.num_agent):
            # physical action space
            u_action_space = spaces.Box(
                low=-np.inf,            #0
                high=+np.inf,          #2
                shape=(self.signal_action_dim,),
                dtype=np.float32,
            )

            if self.movable:
                total_action_space.append(u_action_space)

            # total action space
            self.action_space.append(total_action_space[0])

            # observation space
            share_obs_dim += self.signal_obs_dim
            self.observation_space.append(
                spaces.Box(
                    low=-np.inf,
                    high=+np.inf,
                    shape=(self.signal_obs_dim,),
                    dtype=np.float32,
                )
            )  # [-inf,inf]

        self.share_observation_space = [
            spaces.Box(
                low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32
            )
            for _ in range(self.num_agent)          #?
        ]

    def step(self, actions):
        results = self.env.step(actions)
        obs, rews, dones, infos = results
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        obs = self.env.reset()
        return np.stack(obs)

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass

    def seed(self, seed):
        pass
