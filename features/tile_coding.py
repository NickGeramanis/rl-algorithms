from typing import List, Tuple

import numpy as np
from gym.spaces import Box

from features.feature_constructor import FeatureConstructor


class TileCoding(FeatureConstructor):

    def __init__(self, n_actions: int, n_tilings: int,
                 tiles_per_dimension: List[int], observation_space: Box,
                 displacement_vector: List[float]) -> None:
        self.__n_tilings = n_tilings
        self.__n_actions = n_actions
        self.__tiles_per_dimension = np.array(tiles_per_dimension) + 1
        self.__n_dimensions = len(self.__tiles_per_dimension)
        self.__tiles_per_tiling = np.prod(self.__tiles_per_dimension)
        self.__n_tiles = self.__n_tilings * self.__tiles_per_tiling
        self.n_features = self.__n_tiles * self.__n_actions

        self.__create_tilings(observation_space, displacement_vector)

        self.info = (f'Tile Coding: tilings = {self.__n_tilings},'
                     f'tiles per dimension = {self.__tiles_per_dimension}')

    def __create_tilings(self, observation_space: Box,
                         displacement_vector: List[float]) -> None:
        tile_width = ((observation_space.high - observation_space.low)
                      / self.__tiles_per_dimension)

        tiling_offset = (np.array(displacement_vector)
                         * tile_width / float(self.__n_tilings))

        self.__tilings = np.empty((self.__n_tilings, self.__n_dimensions),
                                  dtype=object)

        minimum_value = observation_space.low
        maximum_value = observation_space.high + tile_width

        # create the first tile
        for dimension_i in range(self.__n_dimensions):
            self.__tilings[0, dimension_i] = np.linspace(
                minimum_value[dimension_i], maximum_value[dimension_i],
                self.__tiles_per_dimension[dimension_i] + 1)

        # subtract an offset from the previous tiling
        # to create the rest.
        for tiling_i in range(1, self.__n_tilings):
            for dimension_i in range(self.__n_dimensions):
                self.__tilings[tiling_i, dimension_i] = (
                    self.__tilings[tiling_i - 1, dimension_i]
                    - tiling_offset[dimension_i])

    def __get_active_features(self, variable: np.ndarray) -> Tuple[int, ...]:
        indices = np.zeros((self.__n_tilings,), object)
        dimensions = np.append(self.__n_tilings, self.__tiles_per_dimension)

        for tiling_i in range(self.__n_tilings):
            index = (tiling_i,)
            for dimension_i in range(self.__n_dimensions):
                index += (np.digitize(
                    variable[dimension_i],
                    self.__tilings[tiling_i, dimension_i]) - 1,)

            for i in range(len(dimensions)):
                indices[tiling_i] += np.prod(dimensions[i + 1:]) * index[i]

        return tuple(indices)

    def calculate_q(self, weights: np.ndarray,
                    state: np.ndarray) -> np.ndarray:
        q = np.empty((self.__n_actions,))
        active_features = self.__get_active_features(state)
        for action in range(self.__n_actions):
            q[action] = np.sum(
                weights[action * self.__n_tiles + active_features])
        return q

    def get_features(self, state: np.ndarray, action: int) -> np.ndarray:
        active_features = self.__get_active_features(state)

        features = np.zeros((self.n_features,))

        features[action * self.__n_tiles + active_features] = 1
        return features
