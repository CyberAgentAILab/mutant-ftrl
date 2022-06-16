import pandas as pd
import matplotlib.pyplot as plt

class FTRLTrajectoryLogger:

    def __init__(self, dir_path, p_id):
        self.dir_path = dir_path
        self.p_id = p_id
    
    def write_trajectories(self, trajectories, index):
        for i_t, trajectory in enumerate(trajectories):
            df = pd.DataFrame(trajectory, index=index)
            df.to_csv('{}/csv/seed_{}_trajectory_{}.csv'.format(self.dir_path, self.p_id, i_t))
            df.plot(alpha = 0.8); 
            plt.ylim(-0.05 ,1.05)
            plt.title('Strategy')
            plt.savefig("{}/figure/seed_{}_trajectory_{}.png".format(self.dir_path, self.p_id, i_t))
            plt.clf()
            plt.close()

    def write_time_avarage_trajectoies(self, trajectories, index):
        for i_t, trajectory in enumerate(trajectories):
            df = pd.DataFrame(trajectory, index=index)
            df.to_csv('{}/csv/seed_{}_time_average_trajectory_{}.csv'.format(self.dir_path, self.p_id, i_t))
            df.plot(alpha = 0.8); 
            plt.ylim(-0.05 ,1.05)
            plt.title('Time Average Strategy')
            plt.savefig("{}/figure/seed_{}_time_average_trajectory_{}.png".format(self.dir_path, self.p_id, i_t))
            plt.clf()
            plt.close()

    def write_exploitabilities(self, exploitabilities, index):
        df = pd.DataFrame(exploitabilities, index=index)
        df.to_csv('{}/csv/seed_{}_exploitability.csv'.format(self.dir_path, self.p_id))
        df.plot()
        plt.title('Exploitability')
        plt.yscale("log")
        plt.savefig("{}/figure/seed_{}_exploitability.png".format(self.dir_path, self.p_id))
        plt.clf()
        plt.close()
    
    def write_time_average_exploitabilities(self, exploitabilities, index):
        df = pd.DataFrame(exploitabilities, index=index)
        df.to_csv('{}/csv/seed_{}_time_average_exploitability.csv'.format(self.dir_path, self.p_id))
        df.plot()
        plt.title('Time Average Exploitability')
        plt.yscale("log")
        plt.savefig("{}/figure/seed_{}_time_average_exploitability.png".format(self.dir_path, self.p_id))
        plt.clf()
        plt.close()