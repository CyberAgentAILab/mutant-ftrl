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

    # run each trial
    logger = Logger()
    for t in np.arange(T + 1):
        if feedback == 'full':
            policies = [player.strategy for player in players]
            utilities = game.full_feedback(policies)
            for i_a, player in enumerate(players):
                player.update(utilities[i_a])
        elif feedback == 'bandit':
            policies = [player.strategy for player in players]
            utilities, actions = game.bandit_feedback(policies)
            for i_a, player in enumerate(players):
                player.update_bandit(utilities[i_a], actions[i_a])
        else:
            raise RuntimeError('illegal feedback type')
        for p in range(len(players)):
            logger['player{}_strategy'.format(p)].append(players[p].strategy.copy())
            logger['player{}_average_strategy'.format(p)].append(players[p].average_iterate_strategy.copy())
        average_iterate_policies = [player.average_iterate_strategy for player in players]
        last_iterate_exploitability = game.calc_exploitability(policies)
        avearge_iterate_exploitability = game.calc_exploitability(average_iterate_policies)
        logger['last_iterate_exploitability'].append(last_iterate_exploitability)
        logger['average_iterate_exploitability'].append(avearge_iterate_exploitability)
        if t % 1000 == 0:
            print('trial: {}, iteration: {}, last-iterate exploitability: {}, average-iterate exploitabiltiy: {}'
                  .format(trial_id, t, last_iterate_exploitability, avearge_iterate_exploitability))
    print('Finish seed {}'.format(trial_id))
    return logger
