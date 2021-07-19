import itertools

import numpy as np

from features.feature_constructor import FeatureConstructor


class Polynomials(FeatureConstructor):

    def __init__(self, n_actions, n_order, n_dimensions):
        self.n_actions = n_actions

        self.n_polynomials = (n_order + 1) ** n_dimensions
        self.n_features = self.n_polynomials * self.n_actions

        self.exponents = list(itertools.product(
            np.arange(n_order + 1), repeat=n_dimensions))

        self.info = f'Polynomials: order = {n_order}'

    def calculate_q(self, weights, state):
        q = np.empty((self.n_actions,))
        for action in range(self.n_actions):
            features = self.get_features(state, action)
            q[action] = np.dot(features, weights)

        return q

    def get_features(self, state, action):
        features = np.zeros((self.n_features,))
        for polynomial_i in range(self.n_polynomials):
            features[action * self.n_polynomials + polynomial_i] = np.prod(
                np.power(state, self.exponents[polynomial_i]))

        return features
