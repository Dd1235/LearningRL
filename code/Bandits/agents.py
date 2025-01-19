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
