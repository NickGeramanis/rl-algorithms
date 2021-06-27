import numpy as np
import math
import random
import logging


class LFASARSA:

    def __init__(self, env, learning_rate_midpoint, discount_factor, initial_learning_rate, learning_rate_steepness, feature_constructor):
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            log_formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
            file_handler = logging.FileHandler('info.log')
            file_handler.setFormatter(log_formatter)
            self.logger.addHandler(file_handler) 
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(log_formatter)
            self.logger.addHandler(console_handler) 
            self.logger.setLevel(logging.INFO)

        self.env = env

        self.discount_factor = discount_factor
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate_steepness = learning_rate_steepness
        self.learning_rate_midpoint = learning_rate_midpoint

        self.feature_constructor = feature_constructor
        self.weights = np.random.random((self.feature_constructor.n_features,))

        self.logger.info('SARSA with Linear Function Approximation: discount factor = {}, learning rate midpoint = {}, learning rate steepness = {}, initial learning rate = {}'.format(
            self.discount_factor, self.learning_rate_midpoint, self.learning_rate_steepness, self.initial_learning_rate))
        self.logger.info(self.feature_constructor.info)

    def train(self, training_episodes):
        for episode_i in range(training_episodes):
            episode_reward = 0.0
            episode_actions = 0

            try:
                learning_rate = self.initial_learning_rate / \
                    (1 + math.exp(self.learning_rate_steepness *
                                  (episode_i - self.learning_rate_midpoint)))
            except OverflowError:
                learning_rate = 0

            try:
                epsilon = 1.0 / (episode_i + 1)
            except OverflowError:
                epsilon = 0

            done = False
            current_state = self.env.reset()
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
                    td_target = reward + self.discount_factor * \
                        next_q_values[next_action]

                td_error = td_target - current_q_values[current_action]

                self.weights += learning_rate * td_error * \
                    self.feature_constructor.get_features(
                        current_state, current_action)

                current_state = next_state
                current_action = next_action
                current_q_values = next_q_values

            self.logger.info('episode={}|reward={}|actions={}'.format(
                episode_i, episode_reward, episode_actions))

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

            self.logger.info('episode={}|reward={}|actions={}'.format(
                episode_i, episode_reward, episode_actions))
