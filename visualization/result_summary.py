import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd
import numpy as np
import ternary
import nashpy
import collections


def calc_lines_end_points(utility):
    lines_end_points = []
    denominator = (-utility[0][0] + utility[0][1] + utility[1][0] - utility[1][1])
    a_1 = (utility[1][0] - utility[1][1] - utility[2][0] + utility[2][1]) / denominator
    a_2 = (-utility[0][0] + utility[0][1] + utility[2][0] - utility[2][1]) / denominator
    b_1 = (utility[1][0] - utility[1][1]) / denominator
    b_2 = (-utility[0][0] + utility[0][1]) / denominator
    y = (-utility[0][1] + utility[1][1]) / (-denominator)

    # x_3 == 0
    if (b_1 >= 0 and b_1 <= 1) and (b_2 >= 0 and b_2 <= 1):
        lines_end_points.append((np.array([b_1, b_2, 0]), np.array([y, 1 - y])))
    # x_1 == 0
    point = b_1 / a_1
    if (point >= 0 and point <= 1):
        lines_end_points.append((np.array([0, 1 - point, point]), np.array([y, 1 - y])))

    # x_2 == 0
    point = b_2 / a_2
    if (point >= 0 and point <= 1):
        lines_end_points.append((np.array([1 - point, 0, point]), np.array([y, 1 - y])))

    assert len(lines_end_points) == 2
    return lines_end_points


def calc_equilibrium_point(utility):
    game = nashpy.Game(utility, utility * -1)
    equilibrias = game.support_enumeration()
    for eq in equilibrias:
        return eq


def triangle_plot(trajectory_value, file_name, title, equilibrium_point=None, lines_ends_point=None):
    # df.plot(alpha = 0.8);
    ###Triangle Plot###
    figure, tax = ternary.figure(scale=1.0)
    # Draw Boundary and Gridlines
    tax.boundary(linewidth=2.0)
    tax.gridlines(color="blue", multiple=0.2)
    # Set Axis labels and Title
    fontsize = 12
    offset = 0.14
    tax.set_title(title, fontsize=fontsize)
    tax.right_corner_label("x1", fontsize=fontsize)
    tax.top_corner_label("x2", fontsize=fontsize)
    tax.left_corner_label("x3", fontsize=fontsize)
    # Plot line of 混合戦略均衡
    tax.plot(trajectory_value, linewidth=2.0, alpha=0.75)
    tax.scatter([trajectory_value[0]], marker='o', color='blue', label="Start Point")
    tax.scatter([trajectory_value[-1]], marker='o', color='red', label="End Point")

    if lines_ends_point:
        p1 = lines_ends_point[0]
        p2 = lines_ends_point[1]
        tax.line(p1, p2, linewidth=2., marker='s', color='green', linestyle=":")
    elif not equilibrium_point is None:
        tax.scatter([equilibrium_point], marker='D', color='black', label="Equilibrium Point")
    # Plot curves
    tax.ticks(axis='lbr', multiple=0.2, linewidth=1, tick_formats="%.1f")
    tax.legend()
    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()
    tax.savefig(file_name + '.pdf')
    plt.clf()
    plt.close()


def triangle_plot_per_seed(dir_path, p_id, utility):
    lines_end_points = None
    if utility.shape[0] == utility.shape[1]:
        equilibrium_points = calc_equilibrium_point(utility)
    elif utility.shape[0] == 3 and utility.shape[1] == 2:
        lines_end_points = calc_lines_end_points(utility)
        equilibrium_points = (lines_end_points[0][0] + lines_end_points[1][0]) / 2, lines_end_points[0][1]
    elif utility.shape[0] == 2 and utility.shape[1] == 3:
        lines_end_points = calc_lines_end_points(utility.T)
        equilibrium_points = (lines_end_points[0][0] + lines_end_points[1][0]) / 2, lines_end_points[0][1]
    else:
        print('invaild utility matrix')
        return
    for i_s in range(2):
        if utility.shape[i_s] == 3 and equilibrium_points:
            lines_end_point = None
            if lines_end_points:
                lines_end_point = [lines_end_points_[i_s] for lines_end_points_ in lines_end_points]
            state_file = glob.glob('{}/csv/seed_{}_time_average_trajectory_{}.csv'.format(dir_path, p_id, i_s))[0]
            df_state = pd.read_csv(state_file, index_col=0)
            triangle_plot(df_state.values,
                          "{}/figure/seed_{}_time_average_trajectories_triangle_{}".format(dir_path, p_id, i_s),
                          "Time Average Trajectory", equilibrium_points[i_s], lines_end_point)
            state_time_average_file = glob.glob('{}/csv/seed_{}_trajectory_{}.csv'.format(dir_path, p_id, i_s))[0]
            df_state_time_average = pd.read_csv(state_time_average_file, index_col=0)
            triangle_plot(df_state_time_average.values,
                          "{}/figure/seed_{}_trajectories_triangle_{}".format(dir_path, p_id, i_s), "Trajectory",
                          equilibrium_points[i_s], lines_end_point)


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
