from features.feature_constructor import FeatureConstructor
import itertools
import math

import numpy as np
from gym.spaces import Box


class FourierBasis(FeatureConstructor):

    def __init__(self, n_actions: int, n_order: int,
                 observation_space: Box) -> None:
        self.__observation_space = observation_space
        self.__n_actions = n_actions
        n_dimensions = len(self.__observation_space.high)
        self.__n_functions = (n_order + 1) ** n_dimensions
        self.n_features = self.__n_functions * self.__n_actions

        self.integer_vector = list(itertools.product(
            np.arange(n_order + 1), repeat=n_dimensions))

        self.info = f'Fourier Basis: order = {n_order}'

    def calculate_q(self, weights: np.ndarray,
                    state: np.ndarray) -> np.ndarray:
        q = np.empty((self.__n_actions,))
        for action in range(self.__n_actions):
            features = self.get_features(state, action)
            q[action] = np.dot(features, weights)

        return q

    def get_features(self, state: np.ndarray, action: int) -> np.ndarray:
        features = np.zeros((self.n_features,))

        norm_state = self.__normalize(state)
        for function_i in range(self.__n_functions):
            features[action * self.__n_functions + function_i] = math.cos(
                math.pi
                * np.dot(norm_state, self.integer_vector[function_i]))

        return features

    def __normalize(self, value: float) -> float:
        return ((value - self.__observation_space.low)
                / (self.__observation_space.high
                   - self.__observation_space.low))
