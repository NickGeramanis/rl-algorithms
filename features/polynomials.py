import itertools

import numpy as np

from features.feature_constructor import FeatureConstructor


class Polynomials(FeatureConstructor):

    def __init__(self, n_actions: int, n_order: int,
                 n_dimensions: int) -> None:
        self.__n_actions = n_actions

        self.__n_polynomials = (n_order + 1) ** n_dimensions
        self.n_features = self.__n_polynomials * self.__n_actions

        self.__exponents = list(itertools.product(
            np.arange(n_order + 1), repeat=n_dimensions))

        self.info = f'Polynomials: order = {n_order}'

    def calculate_q(self, weights: np.ndarray,
                    state: np.ndarray) -> np.ndarray:
        q = np.empty((self.__n_actions,))
        for action in range(self.__n_actions):
            features = self.get_features(state, action)
            q[action] = np.dot(features, weights)

        return q

    def get_features(self, state: np.ndarray, action: int) -> np.ndarray:
        features = np.zeros((self.n_features,))
        for polynomial_i in range(self.__n_polynomials):
            features[action * self.__n_polynomials + polynomial_i] = np.prod(
                np.power(state, self.__exponents[polynomial_i]))

        return features
