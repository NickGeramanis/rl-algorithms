import numpy as np
import math
import random
import logging


class TabularQLearning:

    def __init__(self, env, learning_rate_midpoint, discount_factor, initial_learning_rate, learning_rate_steepness, discretizer):
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            log_formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
            file_handler = logging.FileHandler('info2.log')
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
        self.discretizer = discretizer

        self.q_table = np.random.random(
            (self.discretizer.n_bins + (self.env.action_space.n,)))
        
        self.logger.info('Tabular Q-Learning: discount factor = {}, learning rate midpoint = {}, learning rate steepness = {}, initial learning rate = {}'.format(
            self.discount_factor, self.learning_rate_midpoint, self.learning_rate_steepness, self.initial_learning_rate))
        self.logger.info(self.discretizer.info)

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
            observation = self.env.reset()
            current_state = self.discretizer.get_state(observation)

            while not done:
                if random.random() <= epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[current_state])

                observation, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_actions += 1
                next_state = self.discretizer.get_state(observation)

                if done:
                    td_target = reward
                else:
                    td_target = reward + self.discount_factor * \
                        max(self.q_table[next_state])

                td_error = td_target - self.q_table[current_state + (action,)]
                self.q_table[current_state +
                             (action,)] += learning_rate * td_error

                current_state = next_state

            self.logger.info('episode={}|reward={}|actions={}'.format(
                episode_i, episode_reward, episode_actions))

    def run(self, episodes, render=False):
        for episode_i in range(episodes):
            episode_reward = 0.0
            episode_actions = 0
            observation = self.env.reset()
            done = False

            while not done:
                if render:
                    self.env.render()

                state = self.discretizer.get_state(observation)
                action = np.argmax(self.q_table[state])
                observation, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_actions += 1

            self.logger.info('episode={}|reward={}|actions={}'.format(
                episode_i, episode_reward, episode_actions))
