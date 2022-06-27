import os
import subprocess
import warnings
import multiprocessing as mp
import numpy as np
import random
import glob


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


def load_utility_matrix(game, num_trials):
    if 'random_utility' in game:
        utilities = []
        for file in glob.glob("utility/{}/*".format(game)):
            utilities.append(np.loadtxt(file, delimiter=','))
    else:
        utility = np.loadtxt('utility/{}.csv'.format(game), delimiter=',')
        utilities = [utility] * num_trials
    return utilities


def save_utility_matrix(file_name, utility):
    np.savetxt(file_name, utility, fmt='%.8f', delimiter=',')
