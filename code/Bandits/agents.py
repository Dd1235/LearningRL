import numpy as np
from actors import BaseActor
from estimators import BaseEstimator


class ActionValueBanditAgent:
    """
    Generic agent that holds:
      - a list of estimators (one per arm),
      - an actor that picks an action based on current Q-value estimates.
    """

    def __init__(self, estimators: list[BaseEstimator], actor: BaseActor):
        self.estimators = estimators
        self.actor = actor

    def update(self, action: int, reward: float) -> None:
        """
        Update the estimator corresponding to the chosen action
        with the observed reward.
        """
        self.estimators[action].update(reward)

    def get_estimates(self) -> np.ndarray:
        """
        Return current Q-value estimates as a numpy array, shape (k,).
        """
        return np.array([est.Qn for est in self.estimators])

    def action(self) -> int:
        """
        1) Gather Qn for all arms,
        2) Ask the actor to pick an action given the entire estimate vector,
        3) Return the chosen action (integer index).
        """
        estimates = self.get_estimates()
        return self.actor.select_action(estimates)


# agents.py

import numpy as np


class MedianEliminationAgent:
    """
    Median Elimination agent for stationary multi-armed bandit problems.
    Finds an epsilon-optimal arm with probability >= (1 - delta).
    """

    def __init__(
        self, n_arms: int, epsilon: float, delta: float, random_seed: int = 42
    ):
        """
        Args:
            n_arms (int): Number of bandit arms.
            epsilon (float): Desired accuracy for the final chosen arm (within epsilon of the best arm).
            delta (float): Confidence parameter; we succeed with probability (1 - delta).
            random_seed (int): Random seed for reproducibility.
        """
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.delta = delta
        self.random_seed = random_seed

        self.rng = np.random.default_rng(self.random_seed)
        self.best_arm = None  # Will be determined after train()

    def train(self, environment) -> int:
        """
        Runs the Median Elimination algorithm in a batch style.
        environment must implement:
            - pull_arm(arm_id: int) -> float (returns reward)

        Returns:
            int: The index of the arm estimated to be within epsilon of the true best arm.
        """
        # Start with all arms in the "active set"
        active_arms = np.arange(self.n_arms)
        epsilon_current = self.epsilon
        delta_current = self.delta

        # While we have more than one arm in the active set
        while len(active_arms) > 1:
            # Determine sample size for the current stage
            #    This is from standard Median Elimination analysis:
            #      num_samples = 4/(epsilon_current^2) * ln(3/delta_current)
            #    We'll do a minimal clamp in case logs or denominators cause small or large anomalies in practice.
            num_samples = max(
                1,
                int(
                    np.ceil(
                        4.0
                        / (epsilon_current * epsilon_current)
                        * np.log(3.0 / delta_current)
                    )
                ),
            )

            # Sample each active arm num_samples times and estimate mean reward
            mean_estimates = np.zeros(len(active_arms))
            for i, arm_id in enumerate(active_arms):
                rewards = [environment.pull_arm(arm_id) for _ in range(num_samples)]
                mean_estimates[i] = np.mean(rewards)

            # Determine the median of the estimated means
            median_estimate = np.median(mean_estimates)

            # Eliminate arms whose mean estimate is below the median
            #    i.e., keep only arms with estimated means >= median
            #    or equivalently, discard those strictly < median (ties can go either way).
            keep_mask = mean_estimates >= median_estimate
            active_arms = active_arms[keep_mask]
            mean_estimates = mean_estimates[keep_mask]

            # Update epsilon and delta for the next stage
            #    Typically, we reduce epsilon by factor of 1/2, delta by 1/2
            epsilon_current /= 2.0
            delta_current /= 2.0

            # If we have only one arm left, we can exit early
            if len(active_arms) == 1:
                break

        # We either have exactly one arm left or a small set—pick the last standing
        self.best_arm = active_arms[0]
        return self.best_arm

    def get_action(self) -> int:
        """
        Return the chosen (best) arm after training.
        If training is not done, best_arm is None.
        """
        if self.best_arm is None:
            raise ValueError("Call train(environment) before getting action.")
        return self.best_arm
