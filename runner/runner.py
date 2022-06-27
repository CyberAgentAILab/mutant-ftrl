import numpy as np

from games.matrix_game import MatrixGame
from runner.plotter import Plotter
from runner import utils
from runner.logger import Logger


def run_ftrl(trial_id, game, T, feedback, alg, params, dir_name):
    # set random seed
    utils.set_random_seed(trial_id)

    # initialize game and players
    game = MatrixGame(utils.load_utility_matrix(game, trial_id))
    players = [
        alg(game.num_actions(0), **params),
        alg(game.num_actions(1), **params)
    ]

    logger = Logger()
    plotter = Plotter(dir_name, trial_id)
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
            logger['player{}_average_strategy'.format(p)].append(players[p].time_average_policy.copy())
        time_average_policies = [player.time_average_policy for player in players]
        logger['last_iterate_exploitability'].append(game.calc_exploitability(policies))
        logger['time_average_exploitability'].append(game.calc_exploitability(time_average_policies))
        if i_t > 0 and i_t % int(10e5) == 0:
            plotter.write_trajectories([logger['player{}_strategy'.format(p)] for p in range(len(players))])
            plotter.write_time_avarage_trajectoies([logger['player{}_average_strategy'.format(p)] for p in range(len(players))])
            plotter.write_exploitabilities(logger['last_iterate_exploitability'])
            plotter.write_time_average_exploitabilities(logger['time_average_exploitability'])
            if trial_id % 10 == 0:
                print('p_id', trial_id, ":", i_t, "iterations finished.")
    plotter.write_trajectories([logger['player{}_strategy'.format(p)] for p in range(len(players))])
    plotter.write_time_avarage_trajectoies([logger['player{}_average_strategy'.format(p)] for p in range(len(players))])
    plotter.write_exploitabilities(logger['last_iterate_exploitability'])
    plotter.write_time_average_exploitabilities(logger['time_average_exploitability'])
    print('Finish seed {}'.format(trial_id))
    return players
