from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    """Minimal contract the tuner expects."""


    @abstractmethod
    def fit(self, X, y=None):
        """Train the generator on a pandas.DataFrame X (labels optional)."""
        raise NotImplementedError


    @abstractmethod
    def sample(self, n):
        """Return a pandas.DataFrame with n synthetic rows (and label column if supervised)."""
        raise NotImplementedError