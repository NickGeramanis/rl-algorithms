import random

import numpy as np

from rl_algorithms.rl_algorithm import RLAlgorithhm


class TabularMonteCarlo(RLAlgorithhm):

    def __init__(self, env, discount_factor, discretizer):
        RLAlgorithhm.__init__(self)
        self.env = env
        self.discount_factor = discount_factor
        self.discretizer = discretizer
        self.q_table = np.random.random(
            (self.discretizer.n_bins + (self.env.action_space.n,)))
        self.returns = np.empty(
            (self.discretizer.n_bins + (self.env.action_space.n,)),
            dtype=object)

        self.logger.info(
            f'Tabular Monte Carlo: discount factor = {self.discount_factor}')
        self.logger.info(self.discretizer.info)

    def train(self, training_episodes):
        for episode_i in range(training_episodes):
            episode_reward = 0.0
            episode_actions = 0

            try:
                epsilon = 1.0 / (episode_i + 1)
            except OverflowError:
                epsilon = 0.1

            episode_samples = []
            done = False
            observation = self.env.reset()

            while not done:
                state = self.discretizer.get_state(observation)

                if random.random() <= epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state])

                observation, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_actions += 1

                episode_samples.append((state, action, reward))

            # first-visit mc
            return_ = 0
            proccesed_samples = []
            for sample in reversed(episode_samples):
                state = sample[0]
                action = sample[1]
                reward = sample[2]
                return_ = self.discount_factor * return_ + reward

                if (state, action) in proccesed_samples:
                    continue

                proccesed_samples.append((state, action))
                if self.returns[state + (action,)] is None:
                    self.returns[state + (action,)] = [return_]
                else:
                    self.returns[state + (action,)].append(return_)

                self.q_table[state + (action,)] = (
                    np.mean(self.returns[state + (action,)]))

            self.logger.info(f'episode={episode_i}|reward={episode_reward}'
                             f'|actions={episode_actions}')

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

            self.logger.info(f'episode={episode_i}|reward={episode_reward}'
                             f'|actions={episode_actions}')
