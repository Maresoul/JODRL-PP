import numpy as np
import torch
import torch.autograd as Variable


def soft_update(target, source, tau):
	"""
	Copies the parameters from source network (x) to target network (y) using the below update
	y = TAU*x + (1 - TAU)*y
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(
			target_param.data * (1.0 - tau) + param.data * tau
		)


def hard_update(target, source):
	"""
	Copies the parameters from source network to target network
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
	for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(param.data)



# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:

	def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.25,decay=0.999,last_t=0):
		self.action_dim = action_dim
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.decay = decay
		self.last_t = last_t
		self.X = np.ones(self.action_dim) * self.mu

	def reset(self):
		self.X = np.ones(self.action_dim) * self.mu

	def sample(self,t):
		if self.sigma > 0.05 and t != self.last_t :							#加上衰减，随迭代轮次加大，探索空间减少
			self.last_t = t
			self.sigma *= self.decay
		dx = self.theta * (self.mu - self.X)
		dx = dx + self.sigma * np.random.randn(len(self.X))
		self.X = self.X + dx
		return self.X


# use this to plot Ornstein Uhlenbeck random motion
if __name__ == '__main__':
	ou = OrnsteinUhlenbeckActionNoise(1)
	states = []
	for i in range(1000):
		states.append(ou.sample())
	import matplotlib.pyplot as plt

	plt.plot(states)
	plt.show()