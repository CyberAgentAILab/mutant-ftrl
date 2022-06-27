import numpy as np

from games.matrix_game import MatrixGame
from logger import FTRLTrajectoryLogger
from runner import utils


def run_ftrl(trial_id, game, T, feedback, alg, params, dir_name):
    utils.set_random_seed(trial_id)

    # initialize game and players
    game = MatrixGame(utils.load_utility_matrix(game, trial_id))
    players = [
        alg(game.num_actions(0), **params),
        alg(game.num_actions(1), **params)
    ]

    logger = FTRLTrajectoryLogger(dir_name, trial_id)
    trajectories = [[] for _ in players]
    time_average_trajectories = [[] for _ in players]
    exploitabilities = []
    time_average_exploitabilities = []
    index = []
    for i_t in np.arange(0, T + 1):
        if feedback == 'full':
            policies = [agent.policy for agent in players]
            utilities = game.full_feedback(policies)
            for i_a, agent in enumerate(players):
                agent.update(utilities[i_a])
        elif feedback == 'bandit':
            policies = [agent.policy for agent in players]
            utilities, actions = game.bandit_feedback(policies)
            for i_a, agent in enumerate(players):
                agent.update_bandit(utilities[i_a], actions[i_a])
        else:
            raise RuntimeError('illegal feedback type')
        index.append(i_t)
        for i_a, agent in enumerate(players):
            trajectories[i_a].append(agent.policy.copy())
            time_average_trajectories[i_a].append(agent.time_average_policy.copy())
            time_average_policies = [agent.time_average_policy for agent in players]
        exploitabilities.append(game.calc_exploitability(policies))
        time_average_exploitabilities.append(game.calc_exploitability(time_average_policies))
        if i_t > 0 and i_t % int(10e5) == 0:
            logger.write_trajectories(trajectories, index)
            logger.write_time_avarage_trajectoies(time_average_trajectories, index)
            logger.write_exploitabilities(exploitabilities, index)
            logger.write_time_average_exploitabilities(time_average_exploitabilities, index)
            if trial_id % 10 == 0:
                print('p_id', trial_id, ":", i_t, "iterations finished.")
    logger.write_trajectories(trajectories, index)
    logger.write_time_avarage_trajectoies(time_average_trajectories, index)
    logger.write_exploitabilities(exploitabilities, index)
    logger.write_time_average_exploitabilities(time_average_exploitabilities, index)
    print('Finish seed {}'.format(trial_id))
    return players
