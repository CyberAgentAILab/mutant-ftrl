import numpy as np

from algorithms import FTRL


class MFTRL(FTRL):
    def __init__(self, eta, n_actions, mu, update_freq=0):
        super().__init__(eta, n_actions)
        self.mu = mu
        self.t = 0
        self.mu_policy = np.ones(n_actions) / n_actions
        self.q_value = np.zeros(n_actions)
        self.alpha = 0.01
        self.update_freq = update_freq

    def update(self, utility):
        values = np.exp(self.eta * (utility + self.mu / self.policy * (self.mu_policy - self.policy))) * self.policy
        if self.t > 0 and self.update_freq > 0 and self.t % self.update_freq == 0:
            self.mu_policy = self.policy.copy()
        self.t += 1
        self.policy = values / values.sum()
        self.sum_policy += self.policy
        self.time_average_policy = self.sum_policy / self.sum_policy.sum()

    def update_bandit(self, utility, action):
        values = np.exp(
            self.eta * (utility / self.policy[action] + (self.mu / self.policy) * (self.mu_policy - self.policy))
        ) * self.policy
        if self.t > 0 and self.update_freq > 0 and self.t % self.update_freq == 0:
            self.mu_policy = self.policy.copy()
        self.t += 1
        self.policy = values / values.sum()
        self.sum_policy += self.policy
        self.time_average_policy = self.sum_policy / self.sum_policy.sum()
