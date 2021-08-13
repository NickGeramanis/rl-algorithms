import math
import random

import numpy as np
from features.discretizer import Discretizer
from gym import Env

from rl_algorithms.rl_algorithm import RLAlgorithhm


class TabularQLearning(RLAlgorithhm):

    def __init__(self, env: Env, learning_rate_midpoint: int,
                 discount_factor: float, initial_learning_rate: float,
                 learning_rate_steepness: float,
                 discretizer: Discretizer) -> None:
        RLAlgorithhm.__init__(self)
        self.__env = env
        self.__discount_factor = discount_factor
        self.__initial_learning_rate = initial_learning_rate
        self.__learning_rate_steepness = learning_rate_steepness
        self.__learning_rate_midpoint = learning_rate_midpoint
        self.__discretizer = discretizer
        self.q_table = np.random.random(
            (self.__discretizer.n_bins + (self.__env.action_space.n,)))

        self._logger.info(
            'Tabular Q-Learning:'
            f'discount factor = {self.__discount_factor},'
            f'learning rate midpoint = {self.__learning_rate_midpoint},'
            f'learning rate steepness = {self.__learning_rate_steepness},'
            f'initial learning rate = {self.__initial_learning_rate}')
        self._logger.info(self.__discretizer.info)

    def train(self, training_episodes: int) -> None:
        for episode_i in range(training_episodes):
            episode_reward = 0.0
            episode_actions = 0

            try:
                learning_rate = (
                    self.__initial_learning_rate
                    / (1 + math.exp(
                        self.__learning_rate_steepness
                        * (episode_i - self.__learning_rate_midpoint))))
            except OverflowError:
                learning_rate = 0

            try:
                epsilon = 1.0 / (episode_i + 1)
            except OverflowError:
                epsilon = 0

            done = False
            observation = self.__env.reset()
            current_state = self.__discretizer.get_state(observation)

            while not done:
                if random.random() <= epsilon:
                    action = self.__env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[current_state])

                observation, reward, done, _ = self.__env.step(action)
                episode_reward += reward
                episode_actions += 1
                next_state = self.__discretizer.get_state(observation)

                if done:
                    td_target = reward
                else:
                    td_target = reward + (self.__discount_factor
                                          * max(self.q_table[next_state]))

                td_error = td_target - self.q_table[current_state + (action,)]
                self.q_table[current_state +
                             (action,)] += learning_rate * td_error

                current_state = next_state

            self._logger.info(f'episode={episode_i}|reward={episode_reward}'
                              f'|actions={episode_actions}')

    def run(self, episodes: int, render: bool = False) -> None:
        for episode_i in range(episodes):
            episode_reward = 0.0
            episode_actions = 0
            observation = self.__env.reset()
            done = False

            while not done:
                if render:
                    self.__env.render()

                state = self.__discretizer.get_state(observation)
                action = np.argmax(self.q_table[state])
                observation, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_actions += 1

            self._logger.info(f'episode={episode_i}|reward={episode_reward}'
                              f'|actions={episode_actions}')
