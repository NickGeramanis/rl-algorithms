# Reinforcement Learning Algorithms

This repository provides an implementation for some popular reinforcement learning algorithms that were tested with OpenAI Gym.

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

- [] Policy Iteration
- [] Value Iteration
- [x] Monte Carlo
- [x] SARSA
- [x] Q-Learning
- [x] SARSA(λ)
- [x] Q(λ)
- [x] Monte Carlo with Linear Function Approximation
- [x] SARSA with Linear Function Approximation
- [x] Q-Learning with Linear Function Approximation
- [x] SARSA(λ) with Linear Function Approximation
- [x] Q(λ) with Linear Function Approximation
- [ ] REINFORCE
- [] Actor-Critic with Eligibility Traces
- [] Least-Squares Policy Iteration
- [] Deep Q-Learning

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

In order to test an algorithm, you must create an instance of it with the appropriate arguments.

For example:

```python
env_name = 'MountainCar-v0'
env = gym.make(env_name)

initial_learning_rate = 0.1
learning_rate_steepness = 0.01
learning_rate_midpoint = 1500
discount_factor = 0.99

n_bins = (20, 20)
discrete = False
discretizer = Discretizer(discrete, n_bins, env.observation_space)

tabular_q_learning = TabularQLearning(env, learning_rate_midpoint. discount_factor, initial_learning_rate, learning_rate_steepness, discretizer)
```

And then execute the `train()` method:

```python
training_episodes = 2000
tabular_q_learning.train(training_episodes)
```
![Demonstration](/images/q_learning_mountain_car.gif)

Different algorithms require different arguments. See `agent.py` for more information.


## Status

Under development.


## License

Distributed under the GPL-3.0 License. See `LICENSE` for more information.


## Authors

[Nick Geramanis](https://www.linkedin.com/in/nikolaos-geramanis)

