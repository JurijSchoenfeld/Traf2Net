import pandas as pd
import numpy as np


def map_pid(df):
    # Helper function replacing every p_id with unique index going from 0 to number of unique p_id. 
    pd.options.mode.chained_assignment = None  # default='warn'
    nodes = df.p_id.unique()
    unique_nodes = np.unique(nodes)
    node_int_dict = dict(zip(unique_nodes, np.arange(0, len(unique_nodes), 1)))
    df['p_id'] = df.p_id.map(node_int_dict)

    return df