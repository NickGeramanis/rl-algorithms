from abc import ABC, abstractmethod


class FeatureConstructor(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_features(self, state, action):
        pass

    @abstractmethod
    def calculate_q(self, weights, state):
        pass
