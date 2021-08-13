import random

import numpy as np
from features.discretizer import Discretizer
from gym import Env

from rl_algorithms.rl_algorithm import RLAlgorithhm


class TabularMonteCarlo(RLAlgorithhm):

    def __init__(self, env: Env, discount_factor: float,
                 discretizer: Discretizer):
        RLAlgorithhm.__init__(self)
        self.__env = env
        self.__discount_factor = discount_factor
        self.__discretizer = discretizer
        self.q_table = np.random.random(
            (self.__discretizer.n_bins + (self.__env.action_space.n,)))
        self.returns = np.empty(
            (self.__discretizer.n_bins + (self.__env.action_space.n,)),
            dtype=object)

        self._logger.info(
            f'Tabular Monte Carlo: discount factor = {self.__discount_factor}')
        self._logger.info(self.__discretizer.info)

    def train(self, training_episodes: int) -> None:
        for episode_i in range(training_episodes):
            episode_reward = 0.0
            episode_actions = 0

            try:
                epsilon = 1.0 / (episode_i + 1)
            except OverflowError:
                epsilon = 0.1

            episode_samples = []
            done = False
            observation = self.__env.reset()

            while not done:
                state = self.__discretizer.get_state(observation)

                if random.random() <= epsilon:
                    action = self.__env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state])

                observation, reward, done, _ = self.__env.step(action)
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
                return_ = self.__discount_factor * return_ + reward

                if (state, action) in proccesed_samples:
                    continue

                proccesed_samples.append((state, action))
                if self.returns[state + (action,)] is None:
                    self.returns[state + (action,)] = [return_]
                else:
                    self.returns[state + (action,)].append(return_)

                self.q_table[state + (action,)] = (
                    np.mean(self.returns[state + (action,)]))

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
