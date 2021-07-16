import numpy as np
import itertools
from features.feature_constructor import FeatureConstructor


class RBF(FeatureConstructor):
    def __init__(self, n_actions, observation_space, centers_per_dimension, rbf_standard_deviation):
        self.n_actions = n_actions
        self.observation_space = observation_space

        self.rbf_centers = np.array(
            list(itertools.product(*centers_per_dimension)))

        self.rbf_variance = 2 * rbf_standard_deviation ** 2
        self.n_functions = self.rbf_centers.shape[0] + 1
        self.n_features = self.n_functions * self.n_actions

        self.info = 'Radial Basis Function: centers per dimension = {}, standard deviation = {}'.format(
            centers_per_dimension, rbf_standard_deviation)

    def calculate_q(self, weights, state):
        q = np.empty((self.n_actions,))
        for action in range(self.n_actions):
            features = self.get_features(state, action)
            q[action] = np.dot(features, weights)

        return q

    def get_features(self, state, action):
        features = np.zeros((self.n_features,))

        state = self.normalize(state)
        features[action * self.n_functions] = 1
        for function_i in range(self.rbf_centers.shape[0]):
            features[action * self.n_functions + function_i + 1] = \
                np.exp(-np.linalg.norm(state -
                                       self.rbf_centers[function_i]) ** 2 / self.rbf_variance)

        return features

    def normalize(self, value):
        return (value - self.observation_space.low) / \
               (self.observation_space.high - self.observation_space.low)
