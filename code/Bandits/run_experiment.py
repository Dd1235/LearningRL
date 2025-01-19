import matplotlib.pyplot as plt
import numpy as np
from actors import EpsilonGreedyActor, UCBActor
from agents import ActionValueBanditAgent
from environment import MultiArmNormalBandit
from estimators import (
    ExponentialRecencyWeightedAverageEstimator,
    SampleAverageEstimator,
)


def run_bandit_simulation(env, agent: ActionValueBanditAgent, num_steps: int = 1000):
    rewards = np.zeros(num_steps)
    for t in range(num_steps):

        action = agent.action()
        reward = env.step(action)
        agent.update(action, reward)
        rewards[t] = reward
    return rewards


def create_epsilon_greedy_agent(k=10, epsilon=0.1, seed=123):

    estimators = [SampleAverageEstimator(Q1=0.0) for _ in range(k)]
    actor = EpsilonGreedyActor(k=k, epsilon=epsilon, seed=seed)
    return ActionValueBanditAgent(estimators=estimators, actor=actor)


def create_ucb_agent(k=10, c=2.0, seed=123):

    estimators = [SampleAverageEstimator(Q1=0.0) for _ in range(k)]
    actor = UCBActor(k=k, c=c, seed=seed)
    return ActionValueBanditAgent(estimators=estimators, actor=actor)


def main():

    # create environment, this is all the bandit information
    k = 10
    means = [0, 1, 2, 1.5, 0.5, 1.2, 3, -1, 2.5, 0]
    stds = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # same std for simplicity
    env = MultiArmNormalBandit(means, stds, seed=42)

    # create agents
    eps_agent = create_epsilon_greedy_agent(k=k, epsilon=0.1, seed=123)
    ucb_agent = create_ucb_agent(k=k, c=2.0, seed=123)

    # run simulations
    N = 2000
    rewards_eps = run_bandit_simulation(env, eps_agent, num_steps=N)

    env2 = MultiArmNormalBandit(means, stds, seed=42)
    rewards_ucb = run_bandit_simulation(env2, ucb_agent, num_steps=N)

    avg_rewards_eps = np.cumsum(rewards_eps) / (np.arange(N) + 1)
    avg_rewards_ucb = np.cumsum(rewards_ucb) / (np.arange(N) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(avg_rewards_eps, label="Epsilon-Greedy (eps=0.1)")
    plt.plot(avg_rewards_ucb, label="UCB (c=2.0)")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("Comparison of Epsilon-Greedy vs UCB in 10-Armed Bandit")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
