from __future__ import annotations

from abc import ABC, abstractmethod


class BaseExperimentModel(ABC):
    model_type: str
    input_type: str

    @abstractmethod
    def fit(self, X, y, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError
