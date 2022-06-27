import numpy as np
import pandas as pd

from collections import defaultdict


class Logger(defaultdict):
    def __init__(self, instance_type=list):
        super().__init__(instance_type)

    def to_dataframe(self):
        df = pd.DataFrame()
        for k, v in self.items():
            if isinstance(v[0], (list, np.ndarray)):
                history = np.array(v).T
                for i in range(len(history)):
                    df['{}_{}'.format(k, i)] = history[i]
            else:
                df[k] = v
        return df
