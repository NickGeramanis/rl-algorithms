import numpy as np


class Discretizer:

    def __init__(self, discrete, n_bins, observation_space):
        self.discrete = discrete

        if self.discrete:
            self.n_bins = (observation_space.n,)
            self.info = f'Discrete state space'
            return

        self.n_bins = n_bins
        self.n_dimensions = len(n_bins)

        self.bins = np.empty((self.n_dimensions,), dtype=object)
        for dimension_i in range(self.n_dimensions):
            self.bins[dimension_i] = np.linspace(
                observation_space.low[dimension_i],
                observation_space.high[dimension_i],
                n_bins[dimension_i] + 1)

        self.info = f'Discretizer: bins = {self.bins}'

    def get_state(self, observation):
        if self.discrete:
            return (observation,)

        state = ()
        for dimension_i in range(self.n_dimensions):
            state += (np.digitize(observation[dimension_i],
                                  self.bins[dimension_i]) - 1,)
        return state
