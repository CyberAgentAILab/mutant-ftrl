import collections
import glob
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.ticker import ScalarFormatter


def final_summary(dir_path, n_p):
    final_summary_dict = collections.defaultdict(list)
    exploitability_dfs = []
    time_average_exploitability_dfs = []
    for seed in range(n_p):
        try:
            exploitability_file = glob.glob('{}/csv/seed_{}_exploitability.csv'.format(dir_path, seed))[0]
            time_average_exploitability_file = \
            glob.glob('{}/csv/seed_{}_time_average_exploitability.csv'.format(dir_path, seed))[0]
            exploitability_df = pd.read_csv(exploitability_file, index_col=0)
            time_average_exploitability_df = pd.read_csv(time_average_exploitability_file, index_col=0)
        except:
            print(n_p)
            final_summary_dict['exploitability'].append(None)
            final_summary_dict['time_average_exploitability'].append(None)
            pass
        else:
            exploitability_dfs.append(exploitability_df)
            time_average_exploitability_dfs.append(time_average_exploitability_df)
            final_summary_dict['exploitability'].append(exploitability_df.tail(1).values[0][0])
            final_summary_dict['time_average_exploitability'].append(
                time_average_exploitability_df.tail(1).values[0][0])

    fig, ax = plt.subplots()
    exploitability_df = pd.concat(exploitability_dfs, axis=1)
    smooth_path = exploitability_df.mean(axis='columns')
    smooth_path.to_csv("{}/csv/exploitability_mean.csv".format(dir_path))
    smooth_path.plot(label='Exploitability', alpha=.75)

    time_average_exploitability_df = pd.concat(time_average_exploitability_dfs, axis=1)
    smooth_path = time_average_exploitability_df.mean(axis='columns')
    smooth_path.to_csv("{}/csv/time_average_exploitability_mean.csv".format(dir_path))
    smooth_path.plot(label='Time Average Exploitability', alpha=.75)
    fontsize = 14
    # sns.lineplot(data = header_df[header])
    plt.xlabel('episodes', fontsize=fontsize)
    plt.ylabel('exploitability', fontsize=fontsize)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.xaxis.offsetText.set_fontsize(fontsize)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    plt.yscale('log')
    plt.grid(ls='--')
    plt.legend(fontsize=8)
    plt.tick_params(labelsize=fontsize)
    plt.savefig("{}/figure/exploitability_summary.pdf".format(dir_path))
    plt.clf()
    plt.close()
