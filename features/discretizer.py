from typing import Tuple
from gym.spaces import Box

import numpy as np


class Discretizer:

    def __init__(self, n_bins: Tuple[int, ...],
                 observation_space: Box) -> None:
        self.n_bins = n_bins
        self.__n_dimensions = len(n_bins)

        self.__bins = np.empty((self.__n_dimensions,), dtype=object)
        for dimension_i in range(self.__n_dimensions):
            self.__bins[dimension_i] = np.linspace(
                observation_space.low[dimension_i],
                observation_space.high[dimension_i],
                n_bins[dimension_i] + 1)

        self.info = f'Discretizer: bins = {self.__bins}'

    def get_state(self, observation: np.ndarray) -> Tuple[int, ...]:
        state = ()
        for dimension_i in range(self.__n_dimensions):
            state += (np.digitize(observation[dimension_i],
                                  self.__bins[dimension_i]) - 1,)
        return state
