import numpy as np
from agents import MedianEliminationAgent


class MultiArmedBanditEnv:

    def __init__(self, true_means, random_seed=0):
        """
        Args:
            true_means (np.ndarray): 1D array of length n_arms, containing each arm's true mean reward.
            random_seed (int): for reproducibility.
        """
        self.true_means = np.array(true_means)
        self.rng = np.random.default_rng(seed=random_seed)
        self.n_arms = len(self.true_means)

    def pull_arm(self, arm_id: int) -> float:
        """
        Returns a stochastic reward for pulling the specified arm.
        """
        reward = self.rng.normal(loc=self.true_means[arm_id], scale=1.0)
        return reward


true_means = [0.1, 0.2, 0.5, 0.4, 0.75]

env = MultiArmedBanditEnv(true_means=true_means, random_seed=123)

# We want to find the best arm within epsilon=0.05 with prob >= 1-delta=0.05
agent = MedianEliminationAgent(n_arms=5, epsilon=0.05, delta=0.05, random_seed=123)

best_arm_found = agent.train(env)

print(f"Median Elimination identified arm {best_arm_found+1}th as the best arm.")
print(f"True means: {true_means}")
