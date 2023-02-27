import numpy as np

from algorithms import FTRL


class MFTRL(FTRL):
    def __init__(self, n_actions, random_init_strategy, eta, mu, update_freq=0):
        super().__init__(n_actions, random_init_strategy, eta)
        self.mu = mu
        self.t = 0
        self.ref_strategy = np.ones(n_actions) / n_actions
        self.update_freq = update_freq

    def update(self, utility):
        values = np.exp(self.eta * (utility + self.mu * (self.ref_strategy - self.strategy) / self.strategy)) * self.strategy
        self.strategy = values / values.sum()
        self.t += 1
        self._update_ref_strategy()
        self.sum_strategy += self.strategy
        self.average_iterate_strategy = self.sum_strategy / self.sum_strategy.sum()

    def update_bandit(self, utility, action):
        values = np.exp(
            self.eta * (utility / self.strategy[action] + self.mu * (self.ref_strategy - self.strategy) / self.strategy)
        ) * self.strategy
        self.strategy = values / values.sum()
        self.t += 1
        self._update_ref_strategy()
        self.sum_strategy += self.strategy
        self.average_iterate_strategy = self.sum_strategy / self.sum_strategy.sum()

    def _update_ref_strategy(self):
        if self.update_freq > 0 and self.t % self.update_freq == 0:
            self.ref_strategy = self.strategy.copy()
