import numpy as np

from algorithms.ftrl import FTRL


class OFTRL(FTRL):
    def __init__(self, n_actions, random_initial_strategy, eta):
        super().__init__(n_actions, random_initial_strategy, eta)
        self.prediction_vec = np.zeros(n_actions)
        self.past_utility = np.zeros(n_actions)

    def update(self, utility):
        values = np.exp(self.eta * (2 * utility - self.past_utility)) * self.strategy
        self.strategy = values / values.sum()
        self.past_utility = utility
        self.sum_strategy += self.strategy
        self.average_iterate_strategy = self.sum_strategy / self.sum_strategy.sum()

    def update_bandit(self, utility, action):
        utility[action] = 1 - utility[action]
        values = np.exp(self.eta * (2 * (1 - utility / self.strategy[action]) - self.past_utility)) * self.strategy
        self.past_utility = 1 - utility / self.strategy[action]
        self.strategy = values / values.sum()
        self.sum_strategy += self.strategy
        self.average_iterate_strategy = self.sum_strategy / self.sum_strategy.sum()
