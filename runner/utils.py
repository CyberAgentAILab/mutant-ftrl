import numpy as np


def set_random_seed(seed):
    np.random.seed(seed)


def load_utility_matrix(game, id):
    if 'random_utility' in game:
        return np.loadtxt('utility/{}/utility{}.csv'.format(game, id), delimiter=',')
    else:
        return np.loadtxt('utility/{}.csv'.format(game), delimiter=',')
