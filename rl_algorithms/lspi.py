import numpy as np
from features.feature_constructor import FeatureConstructor
from gym import Env

from rl_algorithms.rl_algorithm import RLAlgorithhm


class LSPI(RLAlgorithhm):

    def __init__(self, env: Env, discount_factor: float,
                 feature_constructor: FeatureConstructor) -> None:
        RLAlgorithhm.__init__(self)
        self.__env = env
        self.__discount_factor = discount_factor
        self.__feature_constructor = feature_constructor
        self.weights = None
        self.sample_set = None

        self._logger.info(f'LSPI: discount factor = {discount_factor}')
        self._logger.info(self.__feature_constructor.info)

    def gather_samples(self, n_samples: int) -> None:
        self.sample_set = np.empty((n_samples,), dtype=object)
        samples_gathered = 0
        done = True

        while samples_gathered < n_samples:
            if done:
                current_state = self.__env.reset()

            action = self.__env.action_space.sample()
            next_state, reward, done, _ = self.__env.step(action)
            self.sample_set[samples_gathered] = (
                current_state, action, reward, next_state, done)
            samples_gathered += 1
            current_state = next_state

    def __calculate_features_list(self) -> np.ndarray:
        features_list = np.empty((self.sample_set.shape[0],), dtype=object)
        sample_i = 0

        for sample in self.sample_set:
            current_state = sample[0]
            action = sample[1]
            features_list[sample_i] = self.__feature_constructor.get_features(
                current_state, action)
            sample_i += 1

        return features_list

    def __lstdq(self, features_list: np.ndarray, delta: float) -> np.ndarray:
        A = delta * np.identity(self.__feature_constructor.n_features)
        b = np.zeros((self.__feature_constructor.n_features,))
        sample_i = 0

        for sample in self.sample_set:
            if sample[4]:
                next_features = np.zeros(
                    (self.__feature_constructor.n_features,))
            else:
                best_action = np.argmax(
                    self.__feature_constructor.calculate_q(
                        self.weights, sample[3]))
                next_features = self.__feature_constructor.get_features(
                    sample[3], best_action)

            if features_list is not None:
                current_features = features_list[sample_i]
                sample_i += 1
            else:
                current_features = self.__feature_constructor.get_features(
                    sample[0], sample[1])

            A += np.outer(
                current_features,
                (current_features - self.__discount_factor * next_features))
            b += current_features * sample[2]

        rank = np.linalg.matrix_rank(A)
        if rank == self.__feature_constructor.n_features:
            A_inverse = np.linalg.inv(A)
        else:
            self._logger.info(f'A is not full rank (rank={rank})')
            u, s, vh = np.linalg.svd(A)
            s = np.diag(s)
            A_inverse = np.matmul(np.matmul(vh.T, np.linalg.pinv(s)), u.T)

        return np.matmul(A_inverse, b)

    def train(self, training_episodes: int, tolerance: float = 0,
              delta: float = 0, pre_calculate_features: bool = False) -> None:
        new_weights = np.random.random(
            (self.__feature_constructor.n_features,))
        if pre_calculate_features:
            features_list = self.__calculate_features_list()
        else:
            features_list = None

        for episode_i in range(training_episodes):
            self.weights = new_weights
            new_weights = self.__lstdq(features_list, delta)

            weights_difference = np.linalg.norm(new_weights - self.weights)
            self._logger.info(f'episode={episode_i}|'
                              f'weights_difference={weights_difference}')

            if weights_difference <= tolerance:
                break

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
