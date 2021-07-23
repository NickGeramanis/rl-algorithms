import math
import random

import numpy as np

from rl_algorithms.rl_algorithm import RLAlgorithhm


class LFASARSALambda(RLAlgorithhm):

    def __init__(self, env, learning_rate_midpoint, discount_factor,
                 initial_learning_rate, learning_rate_steepness,
                 feature_constructor, lambda_):
        RLAlgorithhm.__init__(self)
        self.env = env
        self.lambda_ = lambda_
        self.discount_factor = discount_factor
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate_midpoint = learning_rate_midpoint
        self.learning_rate_steepness = learning_rate_steepness
        self.feature_constructor = feature_constructor
        self.weights = np.random.random((self.feature_constructor.n_features,))

        self.logger.info(
            'SARSA(lambda) with Linear Function Approximation:'
            f'discount factor={self.discount_factor},'
            f' lambda = {self.lambda_},'
            f'learning rate midpoint = {self.learning_rate_midpoint},'
            f'learning rate steepness = {self.learning_rate_steepness},'
            f'initial learning rate = {self.initial_learning_rate}')
        self.logger.info(self.feature_constructor.info)

    def train(self, training_episodes):
        for episode_i in range(training_episodes):
            episode_reward = 0.0
            episode_actions = 0

            try:
                learning_rate = (
                    self.initial_learning_rate
                    / (1 + math.exp(
                        self.learning_rate_steepness
                        * (episode_i - self.learning_rate_midpoint))))
            except OverflowError:
                learning_rate = 0

            try:
                epsilon = 1.0 / (episode_i + 1)
            except OverflowError:
                epsilon = 0

            done = False
            current_state = self.env.reset()
            eligibility_traces = np.zeros(
                (self.feature_constructor.n_features,))
            current_q_values = self.feature_constructor.calculate_q(
                self.weights, current_state)

            if random.random() <= epsilon:
                current_action = self.env.action_space.sample()
            else:
                current_action = np.argmax(current_q_values)

            while not done:
                next_state, reward, done, _ = self.env.step(current_action)
                episode_reward += reward
                episode_actions += 1

                next_q_values = self.feature_constructor.calculate_q(
                    self.weights, next_state)

                if random.random() <= epsilon:
                    next_action = self.env.action_space.sample()
                else:
                    next_action = np.argmax(next_q_values)

                if done:
                    td_target = reward
                else:
                    td_target = reward + (self.discount_factor
                                          * next_q_values[next_action])
                td_error = td_target - current_q_values[current_action]

                eligibility_traces = (
                    self.discount_factor * self.lambda_
                    * eligibility_traces
                    + self.feature_constructor.get_features(
                        current_state, current_action))

                self.weights += learning_rate * td_error * eligibility_traces

                current_state = next_state
                current_action = next_action
                current_q_values = next_q_values

            self.logger.info(f'episode={episode_i}|reward={episode_reward}'
                             f'|actions={episode_actions}')

    def run(self, episodes, render=False):
        for episode_i in range(episodes):
            episode_reward = 0.0
            episode_actions = 0
            state = self.env.reset()
            done = False

            while not done:
                if render:
                    self.env.render()

                action = np.argmax(
                    self.feature_constructor.calculate_q(self.weights, state))

                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_actions += 1

            self.logger.info(f'episode={episode_i}|reward={episode_reward}'
                             f'|actions={episode_actions}')
