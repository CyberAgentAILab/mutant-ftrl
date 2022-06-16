import numpy as np
from scipy.special import softmax

class FollowTheRegularizedLeader:

    def __init__(self, eta, n_actions):
        self.n_actions = n_actions
        self.policy = np.ones(n_actions)/n_actions
        self.eta = eta
        self.sum_policy = np.zeros(n_actions)
        self.time_average_policy = self.policy 

    def update(self, utility):
        exp_utility = np.exp(self.eta * utility)*self.policy
        self.policy = exp_utility/exp_utility.sum()
        self.sum_policy += self.policy 
        self.time_average_policy = self.sum_policy/self.sum_policy.sum()
    
    def update_bandit(self, utility, action):
        utility[action] =  1 - utility[action]
        exp_utility = np.exp(self.eta * (1 - utility/self.policy[action]))*self.policy
        self.policy = exp_utility/exp_utility.sum()
        action = np.random.choice(np.arange(self.n_actions), p=self.policy)
        self.sum_policy += self.policy 
        self.time_average_policy = self.sum_policy/self.sum_policy.sum()

class OptimisticFollowTheRegularizedLeader(FollowTheRegularizedLeader):
    def __init__(self, eta, n_actions):
        super().__init__(eta, n_actions)
        self.prediction_vec = np.zeros(n_actions)
        self.past_utility = np.zeros(n_actions)

    def update(self, utility):
        exp_utility = np.exp(self.eta * (2 * utility - self.past_utility))*self.policy
        self.policy = exp_utility/exp_utility.sum()
        self.past_utility = utility
        self.sum_policy += self.policy 
        self.time_average_policy = self.sum_policy/self.sum_policy.sum()

    def update_bandit(self, utility, action):
        utility[action] =  1 - utility[action]
        exp_utility = np.exp(self.eta * (2 * (1 - utility/self.policy[action]) - self.past_utility))*self.policy
        self.past_utility = 1 - utility/self.policy[action]
        self.policy = exp_utility/exp_utility.sum()
        self.sum_policy += self.policy 
        self.time_average_policy = self.sum_policy/self.sum_policy.sum()

class MutageneticFollowTheRegularizedLeader(FollowTheRegularizedLeader):
    
    def __init__(self, eta, n_actions, mu, update_freq = 0):
        super().__init__(eta, n_actions)
        self.mu = mu
        self.t = 0
        self.mu_policy = np.ones(n_actions)/n_actions
        self.q_value = np.zeros(n_actions)
        self.alpha = 0.01
        self.update_freq = update_freq

    def update(self, utility):
        exp_utility = np.exp(self.eta * (utility + self.mu/self.policy  * ( self.mu_policy - self.policy)))*self.policy
        if self.t > 0 and self.update_freq > 0 and self.t % self.update_freq == 0:
            self.mu_policy = self.policy.copy()
        self.t += 1
        self.policy = exp_utility/exp_utility.sum()
        self.sum_policy += self.policy 
        self.time_average_policy = self.sum_policy/self.sum_policy.sum()
    
    def update_bandit(self, utility, action):
        exp_utility = np.exp(self.eta * (utility/self.policy[action] + (self.mu/self.policy) * ( self.mu_policy - self.policy)))*self.policy
        if self.t > 0 and self.update_freq > 0 and self.t % self.update_freq == 0:
            self.mu_policy = self.policy.copy()
        self.t += 1
        self.policy = exp_utility/exp_utility.sum()
        self.sum_policy += self.policy 
        self.time_average_policy = self.sum_policy/self.sum_policy.sum()
