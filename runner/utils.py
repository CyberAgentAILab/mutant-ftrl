import argparse
import json
import os
import subprocess
import sys
import warnings
import multiprocessing as mp
import numpy as np
import random
import glob


def prepare_output_dir(args, user_specified_dir, argv=None):
    if os.path.exists(user_specified_dir):
        if not os.path.isdir(user_specified_dir):
            raise RuntimeError(
                '{} is not a directory'.format(user_specified_dir))
    outdir = os.path.join(user_specified_dir, args.dir_name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        os.makedirs(outdir + '/csv')
        os.makedirs(outdir + '/figure')
    result_summary_path = "{}/result_summary.txt".format(user_specified_dir)

    with open(result_summary_path, 'a') as f:
        if argv is None:
            argv = sys.argv
        s = '\n' + ' '.join(argv) + ' ' + str(outdir)
        f.write(s)

    # Save all the arguments
    with open(os.path.join(outdir, 'args.txt'), 'w') as f:
        if isinstance(args, argparse.Namespace):
            args = vars(args)
        f.write(json.dumps(args))

    return outdir


def is_return_code_zero(args):
    """Return true iff the given command's return code is zero.
    All the messages to stdout or stderr are suppressed.
    """
    with open(os.devnull, 'wb') as FNULL:
        try:
            subprocess.check_call(args, stdout=FNULL, stderr=FNULL)
        except subprocess.CalledProcessError:
            # The given command returned an error
            return False
        except OSError:
            # The given command was not found
            return False
        return True


class AbnormalExitWarning(Warning):
    """Warning category for abnormal subprocess exit."""
    pass


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def run_async(n_process, run_func):
    """Run experiments asynchronously.
    Args:
      n_process (int): number of processes
      run_func: function that will be run in parallel
    """

    processes = []

    def set_seed_and_run(process_idx, run_func):
        set_random_seed(np.random.randint(0, 2 ** 32))
        run_func(process_idx)

    for process_idx in range(n_process):
        processes.append(mp.Process(target=set_seed_and_run, args=(
            process_idx, run_func)))

    for p in processes:
        p.start()

    for process_idx, p in enumerate(processes):
        p.join()
        if p.exitcode > 0:
            warnings.warn(
                "Process #{} (pid={}) exited with nonzero status {}".format(
                    process_idx, p.pid, p.exitcode),
                category=AbnormalExitWarning,
            )
        elif p.exitcode < 0:
            warnings.warn(
                "Process #{} (pid={}) was terminated by signal {}".format(
                    process_idx, p.pid, -p.exitcode),
                category=AbnormalExitWarning,
            )


def run_async_pool(n_process, run_func, inputs_list):
    multi_pool = mp.Pool(n_process)
    for result in multi_pool.imap_unordered(run_func, inputs_list):
        pass


def get_cpu_count():
    return mp.cpu_count()


def load_payoff_array(payoff_path):
    return np.loadtxt(payoff_path, delimiter=',')


def load_payoff_all_arrays(payoff_path):
    payoffs = []
    for payoff_file in glob.glob("{}/*".format(payoff_path)):
        payoffs.append(np.loadtxt(payoff_file, delimiter=','))
    return payoffs


def save_payoff_array(file_name, payoff):
    np.savetxt(file_name, payoff, fmt='%.8f', delimiter=',')