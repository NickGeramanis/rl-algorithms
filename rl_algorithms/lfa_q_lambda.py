import math
import random

import numpy as np
from features.feature_constructor import FeatureConstructor
from gym import Env

from rl_algorithms.rl_algorithm import RLAlgorithhm


class LFAQLambda(RLAlgorithhm):

    def __init__(self, env: Env, learning_rate_midpoint: int,
                 discount_factor: float, initial_learning_rate: float,
                 learning_rate_steepness: float,
                 feature_constructor: FeatureConstructor,
                 lambda_: float) -> None:
        RLAlgorithhm.__init__(self)
        self.__env = env
        self.__lambda = lambda_
        self.__discount_factor = discount_factor
        self.__initial_learning_rate = initial_learning_rate
        self.__learning_rate_steepness = learning_rate_steepness
        self.__learning_rate_midpoint = learning_rate_midpoint
        self.__feature_constructor = feature_constructor
        self.weights = np.random.random(
            (self.__feature_constructor.n_features,))

        self._logger.info(
            'Q(lambda) with Linear Function Approximation:'
            f'discount factor = {self.__discount_factor},'
            f'lambda = {self.__lambda},'
            f'learning rate midpoint = {self.__learning_rate_midpoint},'
            f'learning rate steepness = {self.__learning_rate_steepness},'
            f'initial learning rate = {self.__initial_learning_rate}')
        self._logger.info(self.__feature_constructor.info)

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
            current_state = self.__env.reset()
            eligibility_traces = np.zeros(
                (self.__feature_constructor.n_features,))
            current_q_values = self.__feature_constructor.calculate_q(
                self.weights, current_state)

            if random.random() <= epsilon:
                current_action = self.__env.action_space.sample()
            else:
                current_action = np.argmax(current_q_values)

            while not done:
                next_state, reward, done, _ = self.__env.step(current_action)
                episode_reward += reward
                episode_actions += 1

                next_q_values = self.__feature_constructor.calculate_q(
                    self.weights, next_state)

                if random.random() <= epsilon:
                    next_action = self.__env.action_space.sample()
                else:
                    next_action = np.argmax(next_q_values)

                if next_q_values[next_action] == np.max(next_q_values):
                    best_action = next_action
                else:
                    best_action = np.argmax(next_q_values)

                if done:
                    td_target = reward
                else:
                    td_target = reward + (self.__discount_factor
                                          * next_q_values[next_action])

                td_error = td_target - current_q_values[current_action]

                if best_action == next_action:
                    eligibility_traces = (
                        self.__discount_factor
                        * self.__lambda * eligibility_traces
                        + self.__feature_constructor.get_features(
                            current_state, current_action))
                else:
                    eligibility_traces = np.zeros(
                        (self.__feature_constructor.n_features,))

                self.weights += learning_rate * td_error * eligibility_traces

                current_state = next_state
                current_action = next_action
                current_q_values = next_q_values

            self._logger.info(f'episode={episode_i}|reward={episode_reward}'
                              f'|actions={episode_actions}')

    def run(self, episodes: int, render: bool = False) -> None:
        for episode_i in range(episodes):
            episode_reward = 0.0
            episode_actions = 0
            state = self.__env.reset()
            done = False

            while not done:
                if render:
                    self.__env.render()

                action = np.argmax(
                    self.__feature_constructor.calculate_q(
                        self.weights, state))
                state, reward, done, _ = self.__env.step(action)
                episode_reward += reward
                episode_actions += 1

            self._logger.info(f'episode={episode_i}|reward={episode_reward}'
                              f'|actions={episode_actions}')
