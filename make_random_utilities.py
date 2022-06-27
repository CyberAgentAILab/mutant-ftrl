import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--size', type=int, default=2, help='size of utility matrix')
parser.add_argument('--n', type=int, default=100, help='number of games')
parser.add_argument('--seed', type=int, default=0, help="random seed")

if __name__ == "__main__":
    args = parser.parse_args()
    np.random.seed(args.seed)
    size = args.size
    outdir = './random_utility/size{}'.format(size)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for i in range(args.n):
        file_name = "utility{}.csv".format(i)
        utility = np.random.rand(size, size)
        np.savetxt('{}/{}'.format(outdir, file_name), utility, fmt='%.8f', delimiter=',')
