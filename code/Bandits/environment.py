import numpy as np


class MultiArmNormalBandit:

    def __init__(self, means, stds, seed=123):
        self.means = means
        self.stds = stds
        self.k = len(means)
        self.rng = np.random.default_rng(seed)

    def step(self, action):
        reward = self.rng.normal(self.means[action], self.stds[action])
        return reward
