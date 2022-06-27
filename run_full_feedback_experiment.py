import argparse
import logger
import visualization
import os
import time

from algorithms import *
from games import matrix_game
from runner import utils
from runner.runner import run_ftrl


def run(T, utilities, alg, params, dir_name, args):
    inputList = [(p_id, T, alg, params, dir_name, args, utilities[p_id]) for p_id in range(args.num_trials)]
    n_pool = min(args.num_trials, int(utils.get_cpu_count() - 1))
    utils.run_async_pool(n_pool, run_exp, inputList)
    visualization.final_summary(dir_name, args.num_trials)


def run_exp(inputs):
    process_idx, T, alg, params, dir_name, args, utility = inputs
    utils.save_utility_matrix('{}/csv/seed_{}_utility.csv'.format(dir_name, process_idx), utility)

    # initialize game and players
    game = matrix_game.MatrixGame(utility)
    players = [
        alg(game.num_actions(0), **params),
        alg(game.num_actions(1), **params)
    ]

    logger_ = logger.FTRLTrajectoryLogger(dir_name, process_idx)
    utils.set_random_seed(process_idx)

    run_ftrl(
        p_id=process_idx,
        game=game,
        agents=players,
        n_iterations=T,
        logger=logger_,
        feedback='full',
        random_policy=args.r_i_p
    )
    print('Finish seed {}'.format(process_idx))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='biased_rsp', type=str,
                        choices=['biased_rps', 'm_eq', *['random_utility/size{}'.format(s) for s in [2, 3, 5, 10, 50, 100]]], help='name of game')
    parser.add_argument('--num_trials', type=int, default=1, help="number of trials to run experiments")
    parser.add_argument('--T', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--r_i_p', '--random_init_policy', action='store_true', help='whether to initialize the initial strategy randomly')
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    args = parser.parse_args()

    # define algorithms
    algs = [
        (FTRL, {'random_initial_policy': args.r_i_p, 'eta': 0.1}),
        (OFTRL, {'random_initial_policy': args.r_i_p, 'eta': 0.1}),
        (MFTRL, {'random_initial_policy': args.r_i_p, 'eta': 0.1, 'mu': 0.01, 'update_freq': 0}),
    ]

    utilities = utils.load_utility_matrix(args.game, args.num_trials)

    for i in range(len(algs)):
        alg = algs[i][0]
        params = algs[i][1]

        dir_name = 'results/{}_feedback/{}/{}'.format('full', args.game, alg.__name__)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            os.makedirs(dir_name + '/csv')
            os.makedirs(dir_name + '/figure')

        utils.set_random_seed(args.seed)

        start = time.time()
        run(args.T, utilities, alg, params, dir_name, args)
        elapsed_time = time.time() - start

        print(dir_name, " elapsed_time:{0}".format(elapsed_time) + "[sec]")


if __name__ == "__main__":
    main()
