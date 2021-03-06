#!/usr/bin/env python
from typing import Tuple

import gym
import numpy as np

from features.discretizer import Discretizer
from features.fourier_basis import FourierBasis
from features.polynomials import Polynomials
from features.rbf import RBF
from features.tile_coding import TileCoding
from rl_algorithms.actor_critic_eligibility_traces import \
    ActorCriticEligibilityTraces
from rl_algorithms.deep_q_learning import DeepQLearning
from rl_algorithms.lfa_monte_carlo import LFAMonteCarlo
from rl_algorithms.lfa_q_lambda import LFAQLambda
from rl_algorithms.lfa_q_learning import LFAQLearning
from rl_algorithms.lfa_sarsa import LFASARSA
from rl_algorithms.lfa_sarsa_lambda import LFASARSALambda
from rl_algorithms.lspi import LSPI
from rl_algorithms.policy_iteration import PolicyIteration
from rl_algorithms.REINFORCE import REINFORCE
from rl_algorithms.tabular_monte_carlo import TabularMonteCarlo
from rl_algorithms.tabular_q_lambda import TabularQLambda
from rl_algorithms.tabular_q_learning import TabularQLearning
from rl_algorithms.tabular_sarsa import TabularSARSA
from rl_algorithms.tabular_sarsa_lambda import TabularSARSALambda
from rl_algorithms.value_iteration import ValueIteration


def main():
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)

    training_episodes = 10
    run_episodes = 10
    weights_filename = 'weights.npy'
    samples_filename = 'track_samples.npy'

    discount_factor = 0.99
    n_dimensions = len(env.observation_space.high)

    ############################################# TABULAR METHODS #############################################
    initial_learning_rate = 0.1
    learning_rate_steepness = 0.01
    learning_rate_midpoint = 1500
    lambda_ = 0.5

    n_bins = (20, 20)
    discretizer = Discretizer(n_bins, env.observation_space)

    tabular_monte_carlo = TabularMonteCarlo(env, discount_factor, discretizer)
    tabular_monte_carlo.train(training_episodes)

    tabular_sarsa = TabularSARSA(
        env, learning_rate_midpoint, discount_factor, initial_learning_rate,
        learning_rate_steepness, discretizer)
    tabular_sarsa.train(training_episodes)

    tabular_q_learning = TabularQLearning(
        env, learning_rate_midpoint, discount_factor, initial_learning_rate,
        learning_rate_steepness, discretizer)
    tabular_q_learning.train(training_episodes)

    tabular_sarsa_lambda = TabularSARSALambda(
        env, learning_rate_midpoint, discount_factor, initial_learning_rate,
        learning_rate_steepness, discretizer, lambda_)
    tabular_sarsa_lambda.train(training_episodes)

    tabular_q_lambda = TabularQLambda(
        env, learning_rate_midpoint, discount_factor, initial_learning_rate,
        learning_rate_steepness, discretizer, lambda_)
    tabular_q_lambda.train(training_episodes)

    ############################################# LFA METHODS #############################################
    tiles_per_dimension = [14, 14]
    displacement_vector = [1, 1]
    n_tilings = 8
    initial_learning_rate = 0.1 / n_tilings
    feature_constructor = TileCoding(
        env.action_space.n, n_tilings, tiles_per_dimension,
        env.observation_space, displacement_vector)
    '''
    n_order = 1
    feature_constructor = Polynomials(
        env.action_space.n, n_order, n_dimensions)

    n_order = 2
    feature_constructor = FourierBasis(
        env.action_space.n, n_order, env.observation_space)
    denominator = np.sum(
        np.power(feature_constructor.integer_vector, 2), axis=1)
    denominator = np.where(denominator == 0, 1, denominator)
    denominator = np.repeat(denominator, env.action_space.n)
    initial_learning_rate = 0.01 / denominator

    rbf_standard_deviation = 0.25
    centers_per_dimension = [
        [0.2, 0.4, 0.6, 0.8],
        [0.2, 0.4, 0.6, 0.8]
    ]
    feature_constructor = RBF(
        env.action_space.n, env.observation_space, centers_per_dimension,
        rbf_standard_deviation)
    '''
    lfa_sarsa = LFASARSA(
        env, learning_rate_midpoint, discount_factor, initial_learning_rate,
        learning_rate_steepness, feature_constructor)
    lfa_sarsa.train(training_episodes)

    lfa_q_learning = LFAQLearning(
        env, learning_rate_midpoint, discount_factor, initial_learning_rate,
        learning_rate_steepness, feature_constructor)
    lfa_q_learning.train(training_episodes)

    lfa_sarsa_lambda = LFASARSALambda(
        env, learning_rate_midpoint, discount_factor, initial_learning_rate,
        learning_rate_steepness, feature_constructor, lambda_)
    lfa_sarsa_lambda.train(training_episodes)

    lfa_q_lambda = LFAQLambda(
        env, learning_rate_midpoint, discount_factor, initial_learning_rate,
        learning_rate_steepness, feature_constructor, lambda_)
    lfa_q_lambda.train(training_episodes)

    tolerance = 0
    delta = 0
    pre_calculate_features = True
    n_samples = 1000
    lspi = LSPI(env, discount_factor, feature_constructor)
    lspi.gather_samples(n_samples)
    # np.save(samples_filename, lspi.sample_set, allow_pickle=True)
    # lspi.sample_set = np.load(samples_filename, allow_pickle=True)
    lspi.train(training_episodes, tolerance, delta, pre_calculate_features)
    lspi.run(run_episodes)


if __name__ == '__main__':
    main()
