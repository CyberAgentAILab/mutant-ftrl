import numpy as np

from games.matrix_game import MatrixGame
from runner import utils
from runner.logger import Logger


def run_ftrl(trial_id, game, T, feedback, alg, params):
    # set random seed
    utils.set_random_seed(trial_id)

    # initialize game and players
    game = MatrixGame(utils.load_utility_matrix(game, trial_id))
    players = [
        alg(game.num_actions(0), **params),
        alg(game.num_actions(1), **params)
    ]

    logger = Logger()
    for i_t in np.arange(T + 1):
        if feedback == 'full':
            policies = [player.policy for player in players]
            utilities = game.full_feedback(policies)
            for i_a, player in enumerate(players):
                player.update(utilities[i_a])
        elif feedback == 'bandit':
            policies = [player.policy for player in players]
            utilities, actions = game.bandit_feedback(policies)
            for i_a, player in enumerate(players):
                player.update_bandit(utilities[i_a], actions[i_a])
        else:
            raise RuntimeError('illegal feedback type')
        for p in range(len(players)):
            logger['player{}_strategy'.format(p)].append(players[p].policy.copy())
            logger['player{}_average_strategy'.format(p)].append(players[p].average_iterate_policy.copy())
        average_iterate_policies = [player.average_iterate_policy for player in players]
        logger['last_iterate_exploitability'].append(game.calc_exploitability(policies))
        logger['average_iterate_exploitability'].append(game.calc_exploitability(average_iterate_policies))
        if i_t > 0 and i_t % int(10e5) == 0:
            if trial_id % 10 == 0:
                print('trial_id', trial_id, ":", i_t, "iterations finished.")
    print('Finish seed {}'.format(trial_id))
    return logger
