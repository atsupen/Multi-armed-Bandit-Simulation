import numpy as np
from scipy.stats import skewnorm
from statistics import stdev, mean
import math
import secrets
import random


# Creating arm_init class for simulation, assuming only normal and skewed-normal distributions
class arm_init:
    def __init__(self, mean, sd, pulls, sample_mean, reward, upper_confidence_bound, sample_sd, skew_factor):
        self.mean = mean
        self.sd = sd
        self.pulls = pulls
        self.sample_mean = sample_mean
        self.reward = reward
        self.ucb = upper_confidence_bound
        self.sample_sd = sample_sd
        self.skew_factor = skew_factor

    def pull(self):
        observed_value = skewnorm.rvs(
            a=self.skew_factor, loc=self.mean, scale=self.sd)
        self.pull_history.append(observed_value)
        self.pulls = len(self.pull_history)
        self.reward += observed_value
        self.sample_mean = self.reward/self.pulls

        if self.pulls <= 1:
            pass

        else:
            self.sample_sd = stdev(self.pull_history)

        return observed_value

    def sample(self, n):
        sample_init = np.random.default_rng()
        observed_value = sample_init.normal(self.mean, self.sd, size=n)
        return observed_value

    def check(self):
        print(
            f"Sample mean: {self.sample_mean}, Total pulls: {self.pulls}, Total reward: {self.reward}")

    def reset_pull_history(self):
        self.pull_history = []


# Calculates the highest expected reward by assuming the optimal strategy is to pull
# the arm with the highest mean.
def get_max_reward(possible_arms, n):
    arm_means = [arm.mean for arm in possible_arms]
    return max(arm_means) * n


def epsilon_greedy(epsilon, n, arm_info, print_result=False):
    possible_arms = [arm_init(*row) for row in arm_info]

    for arm in possible_arms:
        arm.reset_pull_history()

    arm_sample_means = np.array([arm.sample_mean for arm in possible_arms])

    for i in range(n):
        if i == 0:
            chosen_arm = secrets.choice(possible_arms)
            chosen_arm.pull()

        else:
            max_index = np.argmax(arm_sample_means)
            other_arms = np.delete(
                np.array([range(0, (arm_info.shape[0]))]),
                max_index
            )
            possible_options = [max_index,
                                random.choice(other_arms)
                                ]
            chosen_arm = random.choices(
                possible_options, weights=(100 - epsilon, epsilon))
            possible_arms[chosen_arm[0]].pull()

    rewards = [arm.reward for arm in possible_arms]
    best_reward = get_max_reward(possible_arms=possible_arms, n=n)

    if print_result:
        print(f"---Epsilon Greedy---\nHighest Expected Reward: {best_reward}")
        print(
            f"Reward: {sum(rewards):.3f}\nRegret: {best_reward - sum(rewards):.3f}")

    else:
        return sum(rewards), best_reward - sum(rewards)


def epsilon_first(epsilon, n, arm_info, print_result=False):
    possible_arms = [arm_init(*row) for row in arm_info]

    for arm_i in possible_arms:
        arm_i.reset_pull_history()
    arm_sample_means = np.array([arm.sample_mean for arm in possible_arms])

    for i in range(n):
        if i <= epsilon:
            chosen_arm = secrets.choice(possible_arms)
            chosen_arm.pull()

        else:
            max_index = np.argmax(arm_sample_means)
            chosen_arm = possible_arms[max_index]
            chosen_arm.pull()

    rewards = [arm.reward for arm in possible_arms]
    best_reward = get_max_reward(possible_arms=possible_arms, n=n)

    if print_result:
        print(f"---Epsilon First---\nHighest Expected Reward: {best_reward}")
        print(
            f"Reward: {sum(rewards):.3f}\nRegret: {best_reward - sum(rewards):.3f}")

    else:
        return sum(rewards), best_reward - sum(rewards)


# Provides a bonus for uncertainty. This decreases as the arm is pulled more, encouraging
# the exploration of less-pulled arms and the exploitation of more promising ones.
def get_ucb_non_parametric(sample_mean, pulls, sample_size):
    ucb = sample_mean + math.sqrt((2 * math.log(pulls) / sample_size))
    return ucb


def ucb(initial_sample_size, n, arm_info, print_result=False):
    possible_arms = [arm_init(*row) for row in arm_info]
    for arm_i in possible_arms:
        arm_i.reset_pull_history()

    total_pulls = initial_sample_size

    for x in range(arm_info.shape[0]):
        # Sampling each arm a small number of times such that initial ucb's for each arm can be calculated
        arm_x = possible_arms[x]
        for y in range(initial_sample_size):
            arm_x.pull()
    ucb_list = [get_ucb_non_parametric(
        arm_x.sample_mean, initial_sample_size, arm_x.pulls) for arm_x in possible_arms]
    max_ucb_arm = np.argmax(ucb_list)

    for x in range(n - initial_sample_size * arm_info.shape[0]):
        possible_arms[max_ucb_arm].pull()
        total_pulls += 1
        ucb_list[max_ucb_arm] = get_ucb_non_parametric(
            possible_arms[max_ucb_arm].sample_mean, total_pulls, possible_arms[max_ucb_arm].pulls)
        max_ucb_arm = np.argmax(ucb_list)
    rewards = [arm.reward for arm in possible_arms]
    best_reward = get_max_reward(possible_arms=possible_arms, n=n)

    if print_result:
        print(f"---UCB---\nHighest Expected Reward: {best_reward}")
        print(
            f"Reward: {sum(rewards):.3f}\nRegret: {best_reward - sum(rewards):.3f}")

    else:
        return sum(rewards), best_reward - sum(rewards)


# Creates a grid for tuning hyperparameters
def grid(max_value, step):
    grid = list(range(0, max_value + 1, step))
    grid.pop(0)
    return grid


# Allows for model specification and tuning
class model:
    def __init__(self, model_type, number_of_pulls):
        self.parameter = None
        self.number_of_pulls = number_of_pulls
        self.model_type = model_type
        self.arm_info = None

    def select(self, arm_info):
        self.arm_info = arm_info

    def run(self, parameter):
        self.model_type(parameter, self.number_of_pulls)

    def tune(self, grid, tests_per_grid_value):
        results = []

        for x in range(len(grid)):
            grid_value_x_results = []

            for y in range(tests_per_grid_value):
                result_i = (self.model_type(
                    grid[x], tests_per_grid_value, self.arm_info))[0]
                grid_value_x_results.append(result_i)
            results.append(mean(grid_value_x_results))

        max_result_index = results.index(max(results))

        self.parameter = grid[max_result_index]

    def test(self):
        self.model_type(self.parameter, self.number_of_pulls,
                        self.arm_info, print_result=True)


# Setting the properties of each arm (example)
arm_info_1 = np.array([
    # mean, sd, pulls, sample_mean, reward, upper_confidence_bound, sample_sd, skew_factor
    # All except mean, sd and skew_factor (if a skewed norm distribution is desired) should be 0.
    [50, 1, 0, 0, 0, 0, 0, 5],
    [45, 2, 0, 0, 0, 0, 0, 2],
    [46, 1.5, 0, 0, 0, 0, 0, 6],
    [30, 0.5, 0, 0, 0, 0, 0, -3],
    [30, 0.5, 0, 0, 0, 0, 0, -4]
])
