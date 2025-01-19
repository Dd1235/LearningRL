from abc import ABC, abstractmethod

import numpy as np


class BaseActor(ABC):
    @abstractmethod
    def select_action(self, estimates: np.ndarray) -> int:
        """
        Given a full array of Q-value estimates (shape = (k,)),
        return the integer index of the chosen action.
        """
        pass


class EpsilonGreedyActor(BaseActor):
    """
    Epsilon-Greedy actor that chooses a random arm with probability epsilon,
    or an arm among the current best with probability (1 - epsilon).
    """

    def __init__(self, k: int, epsilon: float = 0.1, seed: int = 123):
        self.k = k
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.explore_rng = np.random.default_rng(seed)
        self.explore = False

    def update(self) -> None:
        # Randomly decide if this step is exploration
        self.explore = self.explore_rng.random() < self.epsilon

    def select_action(self, estimates: np.ndarray) -> int:
        """
        1) Possibly explore with probability epsilon.
        2) Else choose an action among the best estimates.
        """
        self.update()

        if self.explore:
            # Explore: pick any arm uniformly
            return self.rng.integers(0, self.k)
        else:
            # Exploit: pick among the best arms
            max_val = np.max(estimates)
            best_actions = np.flatnonzero(estimates == max_val)
            return self.rng.choice(best_actions)


class UCBActor(BaseActor):
    """
    UCB actor that uses the formula:
        a_t = argmax [ Q(a) + c * sqrt(ln(t) / N(a)) ]
    """

    def __init__(self, k: int, c: float = 2.0, seed: int = 123):
        self.k = k
        self.c = c
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(k, dtype=int)  # how many times each arm has been chosen
        self.total_count = 0  # total number of selections so far

    def select_action(self, estimates: np.ndarray) -> int:
        """
        1) For each arm, compute the UCB bonus if it's been selected > 0 times.
        2) Otherwise, unselected arms get an infinite bonus to force exploration.
        """
        self.total_count += 1

        bonus = np.zeros(self.k)
        for a in range(self.k):
            if self.counts[a] == 0:
                bonus[a] = float("inf")  # Force trying each arm at least once
            else:
                bonus[a] = self.c * np.sqrt(np.log(self.total_count) / self.counts[a])

        # Add the bonus to the current Q-value estimates
        ucb_values = estimates + bonus
        best_action = np.argmax(ucb_values)
        self.counts[best_action] += 1
        return best_action
