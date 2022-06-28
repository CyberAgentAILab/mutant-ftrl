import numpy as np


class FTRL(object):
    def __init__(self, n_actions, random_initial_strategy, eta):
        if random_initial_strategy:
            self.strategy = np.random.exponential(scale=1.0, size=n_actions)
            self.strategy /= self.strategy.sum()
        else:
            self.strategy = np.ones(n_actions) / n_actions
        self.eta = eta
        self.sum_strategy = np.zeros(n_actions)
        self.average_iterate_strategy = self.strategy

    def update(self, utility):
        exp_utility = np.exp(self.eta * utility) * self.strategy
        self.strategy = exp_utility / exp_utility.sum()
        self.sum_strategy += self.strategy
        self.average_iterate_strategy = self.sum_strategy / self.sum_strategy.sum()

    def update_bandit(self, utility, action):
        utility[action] = 1 - utility[action]
        exp_utility = np.exp(self.eta * (1 - utility / self.strategy[action])) * self.strategy
        self.strategy = exp_utility / exp_utility.sum()
        self.sum_strategy += self.strategy
        self.average_iterate_strategy = self.sum_strategy / self.sum_strategy.sum()
