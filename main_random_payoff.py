import os
import numpy as np
import argparse
import time



parser = argparse.ArgumentParser()

parser.add_argument('--size', type=int, default=2, help='payoff array size (default: 2)')
parser.add_argument('--n', type=int, default=100, help='number of payoff array(default: 100)')
parser.add_argument('--seed', type=int, default=0, help="random seed")

if __name__ == "__main__":
    args = parser.parse_args()
    np.random.seed(args.seed)
    start = time.time()
    payoff_array_size = args.size
    outdir = './payoff/size_{}'.format(payoff_array_size)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for i in range(args.n):
        file_name = "payoff_{}.csv".format(i)
        np.savetxt('{}/{}'.format(outdir, file_name), np.random.rand(payoff_array_size, payoff_array_size), fmt='%.8f', delimiter=',')