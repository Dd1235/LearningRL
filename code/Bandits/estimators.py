from abc import ABC, abstractmethod


class BaseEstimator(ABC):
    @abstractmethod
    def update(self, reward: float) -> None:
        pass

    @property
    @abstractmethod
    def Qn(self) -> float:
        pass


class SampleAverageEstimator(BaseEstimator):
    def __init__(self, Q1: float):
        self._Qn = Q1
        self.n_updates = 0

    @property
    def Qn(self) -> float:
        return self._Qn

    def update(self, reward: float) -> None:
        self.n_updates += 1
        self._Qn += (reward - self._Qn) / self.n_updates


class ExponentialRecencyWeightedAverageEstimator(BaseEstimator):
    def __init__(self, alpha: float, Q1: float):
        self._Qn = Q1
        self.alpha = alpha

    @property
    def Qn(self) -> float:
        return self._Qn

    def update(self, reward: float) -> None:
        self._Qn += self.alpha * (reward - self._Qn)
