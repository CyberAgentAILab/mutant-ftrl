import numpy as np

from algorithms.ftrl import FTRL


class OFTRL(FTRL):
    def __init__(self, n_actions, random_initial_policy, eta):
        super().__init__(n_actions, random_initial_policy, eta)
        self.prediction_vec = np.zeros(n_actions)
        self.past_utility = np.zeros(n_actions)

    def update(self, utility):
        exp_utility = np.exp(self.eta * (2 * utility - self.past_utility)) * self.policy
        self.policy = exp_utility / exp_utility.sum()
        self.past_utility = utility
        self.sum_policy += self.policy
        self.time_average_policy = self.sum_policy / self.sum_policy.sum()

    def update_bandit(self, utility, action):
        utility[action] = 1 - utility[action]
        values = np.exp(self.eta * (2 * (1 - utility / self.policy[action]) - self.past_utility)) * self.policy
        self.past_utility = 1 - utility / self.policy[action]
        self.policy = values / values.sum()
        self.sum_policy += self.policy
        self.time_average_policy = self.sum_policy / self.sum_policy.sum()
