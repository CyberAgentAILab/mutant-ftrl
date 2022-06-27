import numpy as np


class FTRL(object):
    def __init__(self, n_actions, random_initial_policy, eta):
        if random_initial_policy:
            self.policy = np.random.exponential(scale=1.0, size=n_actions)
            self.policy /= self.policy.sum()
        else:
            self.policy = np.ones(n_actions) / n_actions
        self.eta = eta
        self.sum_policy = np.zeros(n_actions)
        self.time_average_policy = self.policy

    def update(self, utility):
        exp_utility = np.exp(self.eta * utility) * self.policy
        self.policy = exp_utility / exp_utility.sum()
        self.sum_policy += self.policy
        self.time_average_policy = self.sum_policy / self.sum_policy.sum()

    def update_bandit(self, utility, action):
        utility[action] = 1 - utility[action]
        exp_utility = np.exp(self.eta * (1 - utility / self.policy[action])) * self.policy
        self.policy = exp_utility / exp_utility.sum()
        self.sum_policy += self.policy
        self.time_average_policy = self.sum_policy / self.sum_policy.sum()
