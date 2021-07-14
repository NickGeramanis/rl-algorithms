# Reinforcement Learning Algorithms

This repository provides implementation for some popular reinforcement learning algorithms that were tested with OpenAI Gym.

## Table of Contents


- [Description](#description)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Status](#status)
- [License](#license)
- [Authors](#authors)


## Description

The following algorithms have been implemented:

- Policy Iteration
- Value Iteration
- Monte Carlo
- SARSA
- Q-Learning
- SARSA(λ)
- Q(λ)
- Monte Carlo with Linear Function Approximation
- SARSA with Linear Function Approximation
- Q-Learning with Linear Function Approximation
- SARSA(λ) with Linear Function Approximation
- Q(λ) with Linear Function Approximation
- REINFORCE
- Actor-Critic with Eligibility Traces
- Least-Squares Policy Iteration
- Deep Q-Learning

The features in Linear Function Approximation methods can be constructed with the following algorithms:

- Polynomials
- Tile Coding
- Radial Basis Functions
- Fourier Basis


Furthermore, `discretizer.py` implements a method to discretize continuous spaces.

## Getting Started


### Prerequisites

The following libraries need to be installed:

- NumPy
- OpenAI Gym


### Installation

In order to test the algorithms you must import the appropriate package:

For example:

```python
from rl_algorithms.tabular_q_learning import TabularQLearning
```


## Usage

In order to test an algorithm you must create an instance of it with the appropriate arguments.

For example:

```python
env_name = 'MountainCar-v0'
env = gym.make(env_name)

initial_learning_rate = 0.1
learning_rate_steepness = 0.01
learning_rate_midpoint = 1500

n_bins = (20, 20)
discretizer = Discretizer(discrete, n_bins, env.observation_space)

tabular_q_learning = TabularQLearning(env, learning_rate_midpoint. discount_factor, initial_learning_rate, learning_rate_steepness, discretizer)
```

And then execute the `train()` method:

```python
training_episodes = 2000
tabular_q_learning.train(training_episodes)
```

Different algorithms require different arguments. See `agent.py` for more information of how to run an algorithm.


## Status

Under development.


## License

Distributed under the MIT License. See `LICENSE` for more information.


## Authors

[Nick Geramanis](https://www.linkedin.com/in/nikolaos-geramanis)

