import argparse
import os

from algorithms import *
from concurrent.futures import ProcessPoolExecutor
from runner import utils
from runner.runner import run_ftrl


def run_exp(num_trials, game, T, seed, feedback, algs):
    for i in range(len(algs)):
        alg = algs[i][0]
        params = algs[i][1]

        print('==========Run experiments for {}=========='.format(alg.__name__))
        dir_name = 'results/{}_feedback/{}/{}'.format(feedback, game, alg.__name__)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            os.makedirs(dir_name + '/csv')
            os.makedirs(dir_name + '/figure')

        # set random seed
        utils.set_random_seed(seed)

        # run each algorithm
        logs = [None] * num_trials
        with ProcessPoolExecutor() as pool:
            arguments = [[trial_id, game, T, feedback, alg, params] for trial_id in range(num_trials)]
            for trial_id, log in enumerate(pool.map(run_ftrl, *tuple(zip(*arguments)))):
                logs[trial_id] = log

        # save log
        utils.save_and_summary_results(dir_name, logs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='biased_rps', type=str,
                        choices=['biased_rps', 'm_eq', *['random_utility/size{}'.format(s) for s in [2, 3, 5, 10, 50, 100]]], help='name of game')
    parser.add_argument('--num_trials', type=int, default=1, help="number of trials to run experiments")
    parser.add_argument('--T', type=int, default=1000, help='number of iterations')
    parser.add_argument('--feedback', type=str, default='full', choices=['full', 'bandit'], help="feedback type")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--random_init_strategy', action='store_true', help='whether to initialize the initial strategy randomly')
    args = parser.parse_args()

    # define algorithms
    algs = [
        (FTRL, {'random_initial_strategy': args.random_init_strategy, 'eta': 0.1}),
        (OFTRL, {'random_initial_strategy': args.random_init_strategy, 'eta': 0.1}),
        (MFTRL, {'random_initial_strategy': args.random_init_strategy, 'eta': 0.1, 'mu': 0.01, 'update_freq': 0}),
    ]

    # run experiments
    print('==========Run experiment over {} trials=========='.format(args.num_trials))
    run_exp(args.num_trials, args.game, args.T, args.seed, args.feedback, algs)


if __name__ == "__main__":
    main()
