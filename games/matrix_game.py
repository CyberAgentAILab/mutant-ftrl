import numpy as np


class MatrixGame(object):
    def __init__(self, utility):
        self.utility = utility

    def full_feedback(self, strategies):
        return [self.utility @ strategies[1], -self.utility.T @ strategies[0]]

    def bandit_feedback(self, strategies):
        actions = [np.random.choice(np.arange(len(strategies[i])), p=strategies[i]) for i in range(len(strategies))]
        loss = [np.zeros(self.utility.shape[0]), np.zeros(self.utility.shape[1])]
        loss[0][actions[0]] = self.utility[tuple(actions)]
        loss[1][actions[1]] = -self.utility[tuple(actions)]
        return loss, actions

    def calc_exploitability(self, strategies):
        return max(self.utility @ strategies[1]) + max(-self.utility.T @ strategies[0])

    def num_actions(self, player_id):
        return self.utility.shape[player_id]
