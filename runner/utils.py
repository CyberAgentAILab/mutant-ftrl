import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.size'] = 12
pd.options.display.float_format = '{:.3f}'.format
plt.rc('pdf', fonttype=42)
plt.rc('ps', fonttype=42)
plt.rcParams['figure.subplot.left'] = 0.135


def set_random_seed(seed):
    np.random.seed(seed)


def load_utility_matrix(game, id):
    if 'random_utility' in game:
        return np.loadtxt('utility/{}/utility{}.csv'.format(game, id), delimiter=',')
    else:
        return np.loadtxt('utility/{}.csv'.format(game), delimiter=',')


def save_and_summary_results(dir_name, logs):
    # save log
    for j in range(len(logs)):
        df = logs[j].to_dataframe()
        df.index.name = '#index'
        df.to_csv(dir_name + '/csv/seed{}_results.csv'.format(j))

    # calc mean exploitability
    last_iterate_exploitability_dfs = []
    average_iterate_exploitability_dfs = []
    for i in range(len(logs)):
        df = logs[i].to_dataframe()
        last_iterate_exploitability_dfs.append(df['last_iterate_exploitability'])
        average_iterate_exploitability_dfs.append(df['average_iterate_exploitability'])
    exploitability_df = pd.concat(last_iterate_exploitability_dfs, axis=1)
    average_iterate_exploitability_df = pd.concat(average_iterate_exploitability_dfs, axis=1)

    # save mean exploitability
    df = pd.concat([logs[0].to_dataframe()['iteration'],
                    exploitability_df.mean(axis='columns'),
                    average_iterate_exploitability_df.mean(axis='columns')], axis=1)
    df.index.name = '#index'
    df.columns = ['iteration', 'last_iterate_exploitability', 'average_iterate_exploitability']
    df.to_csv('{}/csv/exploitability_mean.csv'.format(dir_name))

    # plot mean exploitability
    plt.plot(df['iteration'], df[['last_iterate_exploitability', 'average_iterate_exploitability']],
             label=['Last-Iterate Exploitability', 'Average-Iterate Exploitability'])
    plt.xlabel('Iterations')
    plt.ylabel('Exploitability')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(ls='--')
    plt.legend()
    plt.savefig('{}/figure/exploitability_mean.pdf'.format(dir_name))
    plt.close()
