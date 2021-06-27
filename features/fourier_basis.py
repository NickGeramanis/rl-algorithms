import numpy as np
import itertools
import math
from features.feature_constructor import FeatureConstructor


class FourierBasis(FeatureConstructor):

    def __init__(self, n_actions, n_order, n_dimensions, observation_space):
        self.observation_space = observation_space
        self.n_actions = n_actions

        self.n_functions = (n_order + 1) ** n_dimensions
        self.n_features = self.n_functions * self.n_actions

        self.integer_vector = list(itertools.product(
            np.arange(n_order + 1), repeat=n_dimensions))
        
        self.info = 'Fourier Basis: order = {}'.format(n_order)

    def calculate_q(self, weights, state):
        q = np.empty((self.n_actions,))
        for action in range(self.n_actions):
            features = self.get_features(state, action)
            q[action] = np.dot(features, weights)

        return q

    def get_features(self, state, action):
        features = np.zeros((self.n_features,))

        norm_state = self.normalize(state)
        for function_i in range(self.n_functions):
            features[action * self.n_functions + function_i] = math.cos(
                math.pi * np.dot(norm_state, self.integer_vector[function_i]))

        return features

    def normalize(self, value):
        return (value - self.observation_space.low) / \
               (self.observation_space.high - self.observation_space.low)
