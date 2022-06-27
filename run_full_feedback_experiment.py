import argparse
import logger
import visualization
import time

from algorithms import *
from games import matrix_game
from runner import utils
from runner.ftrl_runner import run_ftrl

parser = argparse.ArgumentParser()
parser.add_argument('--payoff', default='payoff/biased_rsp.csv', type=str, help='payoff csv file path using single population')
parser.add_argument('--n_p', '--n_processes', type=int, default=1, help="number of process different seed")
parser.add_argument('--n_i', '--n_iterations', type=int, default=1000, help='number of iterations (default: 1000)')
parser.add_argument('--update_freq', '--uf', type=int, default=0, help='update frequency of reference strategy')
parser.add_argument('--arch', type=str, default='mftrl', choices=['ftrl', 'oftrl', 'mftrl'])
parser.add_argument('--mu', type=float, default=0.01, help='mutation rate')
parser.add_argument('--eta', type=float, default=0.1, help='learning rate')
parser.add_argument('--outdir', type=str, default='results', help='Directory path to save output files. If it does not exist, it will be created.')
parser.add_argument('--dir_suffix', type=str, help='Directory suffix to save output files')
parser.add_argument('--r_p', '--random_payoff', action='store_true', help='random payoff')
parser.add_argument('--r_i_p', '--random_init_policy', action='store_true', help='random random policy')
parser.add_argument('--size', type=int, default=2, choices=(2, 3, 5, 10, 50, 100), help='random payoff size')
parser.add_argument('--seed', type=int, default=0, help="random seed")


def main(args):
    if args.r_p:
        payoffs = utils.load_payoff_all_arrays('payoff/size_{}'.format(args.size))
    else:
        payoff = utils.load_payoff_array(args.payoff)
        payoffs = [payoff] * args.n_p

    inputList = [(p_id, args, payoffs[p_id]) for p_id in range(args.n_p)]
    n_pool = min(args.n_p, int(utils.get_cpu_count() - 1))
    utils.run_async_pool(n_pool, run, inputList)
    visualization.final_summary(args.outdir, args.n_p)


def run(inputs):
    process_idx, args, payoff = inputs
    utils.save_payoff_array('{}/csv/seed_{}_payoff.csv'.format(args.outdir, process_idx), payoff)
    game = matrix_game.MatrixGame(payoff)

    if args.arch == 'ftrl':
        agents = [
            FTRL(args.eta, game.num_actions(0)),
            FTRL(args.eta, game.num_actions(1))
        ]
    elif args.arch == 'oftrl':
        agents = [
            OFTRL(args.eta, game.num_actions(0)),
            OFTRL(args.eta, game.num_actions(1))
        ]
    elif args.arch == 'mftrl':
        agents = [
            MFTRL(args.eta, game.num_actions(0), args.mu, update_freq=args.update_freq),
            MFTRL(args.eta, game.num_actions(1), args.mu, update_freq=args.update_freq),
        ]
    else:
        assert False
    logger_ = logger.FTRLTrajectoryLogger(args.outdir, process_idx)
    utils.set_random_seed(process_idx)

    run_ftrl(
        p_id=process_idx,
        game=game,
        agents=agents,
        n_iterations=args.n_i,
        logger=logger_,
        feedback='full',
        random_policy=args.r_i_p
    )
    print('Finish seed {}'.format(process_idx))


if __name__ == "__main__":
    args = parser.parse_args()
    if args.r_p:
        payoff = 'payoff/size_{}'.format(args.size)
    else:
        payoff = args.payoff
        payoff = payoff.replace('.csv', '')
    payoff = payoff.replace('/', '_')
    args.dir_name = 'arch_{}_{}_feedback_{}_{}'.format(args.arch, payoff, 'full', args.dir_suffix)
    args.outdir = utils.prepare_output_dir(args, args.outdir)
    utils.set_random_seed(args.seed)

    start = time.time()
    main(args)
    elapsed_time = time.time() - start

    print(args.outdir, " elapsed_time:{0}".format(elapsed_time) + "[sec]")
