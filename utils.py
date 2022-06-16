import matplotlib.pyplot as plt
import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile
import warnings
import multiprocessing as mp
import numpy as np
import random
import glob

def prepare_output_dir(args, user_specified_dir=None, argv=None,
                       time_format='%Y%m%dT%H%M%S%f'):
    """Prepare a directory for outputting training results.
    An output directory, which ends with the current datetime string,
    is created. Then the following infomation is saved into the directory:
        args.txt: command line arguments
        command.txt: command itself
        environ.txt: environmental variables
    Additionally, if the current directory is under git control, the following
    information is saved:
        git-head.txt: result of `git rev-parse HEAD`
        git-status.txt: result of `git status`
        git-log.txt: result of `git log`
        git-diff.txt: result of `git diff`
    Args:
        args (dict or argparse.Namespace): Arguments to save
        user_specified_dir (str or None): If str is specified, the output
            directory is created under that path. If not specified, it is
            created as a new temporary directory instead.
        argv (list or None): The list of command line arguments passed to a
            script. If not specified, sys.argv is used instead.
        time_format (str): Format used to represent the current datetime. The
        default format is the basic format of ISO 8601.
    Returns:
        Path of the output directory created by this function (str).
    """


    time_str = datetime.datetime.now().strftime(time_format)
    if user_specified_dir is not None:
        if os.path.exists(user_specified_dir):
            if not os.path.isdir(user_specified_dir):
                raise RuntimeError(
                    '{} is not a directory'.format(user_specified_dir))
        dir_name = '{}_{}'.format(time_str, args.dir_name)
        outdir = os.path.join(user_specified_dir, dir_name)
        if os.path.exists(outdir):
            raise RuntimeError('{} exists'.format(outdir))
        else:
            os.makedirs(outdir)
            os.makedirs(outdir + '/csv')
            os.makedirs(outdir + '/figure')
    else:
        outdir = tempfile.mkdtemp(prefix=time_str)
    result_summary_path = "{}/result_summary.txt".format(user_specified_dir)

    with open(result_summary_path, 'a') as f:
        if argv is None:
            argv = sys.argv
        s = '\n' + ' '.join(argv) + ' ' +str(outdir)
        f.write(s)

    # Save all the arguments
    with open(os.path.join(outdir, 'args.txt'), 'w') as f:
        if isinstance(args, argparse.Namespace):
            args = vars(args)
        f.write(json.dumps(args))

    # Save all the environment variables
    with open(os.path.join(outdir, 'environ.txt'), 'w') as f:
        f.write(json.dumps(dict(os.environ)))

    # Save the command
    with open(os.path.join(outdir, 'command.txt'), 'w') as f:
        if argv is None:
            argv = sys.argv
        f.write(' '.join(argv))

    if is_under_git_control():
        # Save `git rev-parse HEAD` (SHA of the current commit)
        with open(os.path.join(outdir, 'git-head.txt'), 'wb') as f:
            f.write(subprocess.check_output('git rev-parse HEAD'.split()))

        # Save `git status`
        with open(os.path.join(outdir, 'git-status.txt'), 'wb') as f:
            f.write(subprocess.check_output('git status'.split()))

        # Save `git log`
        with open(os.path.join(outdir, 'git-log.txt'), 'wb') as f:
            f.write(subprocess.check_output('git log'.split()))

        # Save `git diff`
        with open(os.path.join(outdir, 'git-diff.txt'), 'wb') as f:
            f.write(subprocess.check_output('git diff'.split()))

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

def is_under_git_control():
    """Return true iff the current directory is under git control."""
    return is_return_code_zero(['git', 'rev-parse'])

class AbnormalExitWarning(Warning):
    """Warning category for abnormal subprocess exit."""
    pass

def set_random_seed(seed) :
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

def load_fixed_point(fixed_point_path):
    return np.loadtxt(fixed_point_path, delimiter=',')

def load_payoff_array(payoff_path):
    return np.loadtxt(payoff_path, delimiter=',')

def load_payoff_all_arrays(payoff_path):
    payoffs = []
    for payoff_file in glob.glob("{}/*".format(payoff_path)):
        payoffs.append(np.loadtxt(payoff_file, delimiter=','))
    return payoffs

def save_payoff_array(file_name, payoff):
    np.savetxt(file_name, payoff, fmt='%.8f', delimiter=',')