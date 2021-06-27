#!/usr/bin/env python
import gym
import numpy as np
from rl_algorithms.policy_iteration import PolicyIteration
from rl_algorithms.value_iteration import ValueIteration
from rl_algorithms.tabular_monte_carlo import TabularMonteCarlo
from rl_algorithms.tabular_sarsa import TabularSARSA
from rl_algorithms.tabular_q_learning import TabularQLearning
from rl_algorithms.tabular_sarsa_lambda import TabularSARSALambda
from rl_algorithms.tabular_q_lambda import TabularQLambda
from rl_algorithms.lfa_monte_carlo import LFAMonteCarlo
from rl_algorithms.lfa_sarsa import LFASARSA
from rl_algorithms.lfa_q_learning import LFAQLearning
from rl_algorithms.lfa_sarsa_lambda import LFASARSALambda
from rl_algorithms.lfa_q_lambda import LFAQLambda
from rl_algorithms.tabular_dyna_q import TabularDynaQ
from rl_algorithms.REINFORCE import REINFORCE
from rl_algorithms.actor_critic_eligibility_traces import ActorCriticEligibilityTraces
from rl_algorithms.lspi import LSPI
from rl_algorithms.deep_q_learning import DeepQLearning
from features.discretizer import Discretizer
from features.tile_coding import TileCoding
from features.fourier_basis import FourierBasis
from features.polynomials import Polynomials
from features.rbf import RBF


def main():
    #env_name = 'MountainCar-v0'
    env_name = 'FrozenLake-v0'
    env = gym.make(env_name)

    training_episodes = 2000
    run_episodes = 10
    weights_filename = 'weights.npy'
    samples_filename = 'track_samples.npy'

    discount_factor = 0.99
    if type(env.observation_space) is gym.spaces.Box:
        n_dimensions = env.observation_space.low.size
        discrete = False
    elif type(env.observation_space) is gym.spaces.Discrete:
        discrete = True
    
    ############################################# TABULAR METHODS #############################################
    initial_learning_rate = 0.1
    learning_rate_steepness = 0.01
    learning_rate_midpoint = 1500
    lambda_ = 0.5
    
    n_bins = (20, 20)
    discretizer = Discretizer(discrete, n_bins, env.observation_space)

    tabular_monte_carlo = TabularMonteCarlo(env, discount_factor, discretizer)
    tabular_monte_carlo.train(training_episodes)
    print(tabular_monte_carlo.q_table)
    #tabular_monte_carlo.run(run_episodes, True)
    '''
    tabular_sarsa = TabularSARSA(env, learning_rate_midpoint, discount_factor,
                                 initial_learning_rate, learning_rate_steepness, discretizer)
    tabular_sarsa.train(training_episodes)
    
    tabular_q_learning = TabularQLearning(
        env, learning_rate_midpoint, discount_factor, initial_learning_rate, learning_rate_steepness, discretizer)
    tabular_q_learning.train(training_episodes)
    
    tabular_sarsa_lambda = TabularSARSALambda(
        env, learning_rate_midpoint, discount_factor, initial_learning_rate, learning_rate_steepness, discretizer, lambda_)
    tabular_sarsa_lambda.train(training_episodes)
    
    tabular_q_lambda = TabularQLambda(env, learning_rate_midpoint, discount_factor,
                                      initial_learning_rate, learning_rate_steepness, discretizer, lambda_)
    tabular_q_lambda.train(training_episodes)
    
    ############################################# LFA METHODS #############################################
    tiles_per_dimension = [14, 14]
    displacement_vector = [1, 1]
    n_tilings = 8
    initial_learning_rate = 0.1 / n_tilings
    feature_constructor = TileCoding(
        env.action_space.n, n_tilings, tiles_per_dimension, env.observation_space, displacement_vector)
    
    n_order = 1
    feature_constructor = Polynomials(
        env.action_space.n, n_order, n_dimensions)

    n_order = 2
    feature_constructor = FourierBasis(
        env.action_space.n, n_order, n_dimensions, env.observation_space)
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
        env.action_space.n, env.observation_space, centers_per_dimension, rbf_standard_deviation)
    
    lfa_sarsa = LFASARSA(env, learning_rate_midpoint, discount_factor,
                         initial_learning_rate, learning_rate_steepness, feature_constructor)
    lfa_sarsa.train(training_episodes)

    lfa_q_learning = LFAQLearning(env, learning_rate_midpoint, discount_factor,
                                  initial_learning_rate, learning_rate_steepness, feature_constructor)
    lfa_q_learning.train(training_episodes)
    
    lfa_sarsa_lambda = LFASARSALambda(env, learning_rate_midpoint, discount_factor,
                                      initial_learning_rate, learning_rate_steepness, feature_constructor, lambda_)
    lfa_sarsa_lambda.train(training_episodes)
    #np.save(samples_filename, lfa_sarsa_lambda.sample_set, allow_pickle=True)
    
    lfa_q_lambda = LFAQLambda(env, learning_rate_midpoint, discount_factor,
                              initial_learning_rate, learning_rate_steepness, feature_constructor, lambda_)
    lfa_q_lambda.train(training_episodes)

    tolerance = 0
    delta = 0
    pre_calculate_features = True
    lspi = LSPI(env, discount_factor, feature_constructor)
    # lspi.gather_samples(n_samples)
    lspi.sample_set = np.load(samples_filename, allow_pickle=True)
    lspi.train(training_episodes, delta=delta, pre_calculate_features=pre_calculate_features)
    lspi.run(run_episodes)
    '''

if __name__ == '__main__':
    main()
