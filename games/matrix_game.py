import numpy as np


class MatrixGame(object):
    def __init__(self, payoff):
        self.payoff = payoff

    def full_feedback(self, strategies):
        return [self.payoff @ strategies[1], -self.payoff.T @ strategies[0]]

    def bandit_feedback(self, strategies):
        actions = [np.random.choice(np.arange(len(strategies[i])), p=strategies[i]) for i in range(len(strategies))]
        loss = [np.zeros(self.payoff.shape[0]), np.zeros(self.payoff.shape[1])]
        loss[0][actions[0]] = self.payoff[tuple(actions)]
        loss[1][actions[1]] = -self.payoff[tuple(actions)]
        return loss, actions

    def calc_exploitability(self, strategies):
        return max(self.payoff @ strategies[1]) + max(-self.payoff.T @ strategies[0])

    def num_actions(self, player_id):
        return self.payoff.shape[player_id]
