import pandas as pd
import numpy as np
from math import log10
from collections import Counter
import numpy as np
from itertools import product

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    print("\033[1m tacoma does not install `matplotlib` as a dependency. Please install it manually. \033[0m")
    raise e


from tacoma import marker_sequence
from tacoma import color_sequence
from tacoma import get_logarithmic_histogram


def map_pid(df):
    # Helper function replacing every p_id with unique index going from 0 to number of unique p_id. 
    pd.options.mode.chained_assignment = None  # default='warn'
    nodes = df.p_id.unique()
    unique_nodes = np.unique(nodes)
    node_int_dict = dict(zip(unique_nodes, np.arange(0, len(unique_nodes), 1)))
    df['p_id'] = df.p_id.map(node_int_dict)

    return df


def plot_contact_durations(result,
                           ax,
                           marker=None,
                           xlabel='duration',
                           bins=100,  # number of bins
                           time_normalization_factor=1.,
                           time_unit=None,
                           bin_dt=None,
                           plot_step=False,
                           fit_power_law=False,
                           use_logarithmic_histogram=True,
                           markersize=4,
                           label=None,
                           color=None
                           ):
    if marker is None:
        markers = ['o', 'x']
    else:
        markers = [marker] * 2

    if label is None:
        labels = ['contact', 'inter-contact']
    else:
        # labels = [label] * 2
        labels = label


    if bin_dt is not None:
        use_discrete_dt = True
    else:
        use_discrete_dt = False

    if not hasattr(ax,'__len__'):
        a = [ax, ax]
    else:
        a = ax

    durs = [np.array(result.contact_durations, dtype=float),
            np.array(result.group_durations[1], dtype=float)]
    res = {}

    for i, dur in enumerate(durs):
        if not plot_step:
            if use_logarithmic_histogram:
                x, y = get_logarithmic_histogram(
                    time_normalization_factor*dur, bins)
            elif use_discrete_dt:
                c = Counter(dur / bin_dt)
                total = sum(c.values())
                x = []
                y = []
                for x_, y_ in c.items():
                    x.append(x_* bin_dt)
                    y.append(y_/total / bin_dt)
            else:
                y, x = np.histogram(dur*time_normalization_factor,bins=bins,density=True)
                x = 0.5*(x[1:]+x[:-1])
                print(x.shape,y.shape)

            a[i].plot(x, y, ls='', marker=markers[i], color=color,
                    label=labels[i],
                    ms=markersize,
                    mew=1,
                    mfc='None'
                    )
        else:
            if use_logarithmic_histogram:
                x, y = get_logarithmic_histogram(
                    time_normalization_factor*dur, bins, return_bin_means=False)
            elif use_discrete_dt:
                c = Counter(dur / bin_dt)
                total = sum(c.values())
                x = []
                y = []
                for x_, y_ in c.items():
                    x.append(x_* bin_dt)
                    y.append(y_/total / bin_dt)
                x.append(x[-1]+1)
            else:
                y, x = np.histogram(dur*time_normalization_factor,bins=bins,density=True)
            a[i].step(x, np.append(y, y[-1]),
                    where='post',
                    label=labels[i],
                    color=color
                    )

        res[labels[i]] = (x, y)

    if time_unit is not None:
        xlabel += ' [' + time_unit + ']'
    ylabel = 'probability density'
    if time_unit is not None:
        ylabel += ' [1/' + time_unit + ']'

    for ax in a: 
 
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()

    return res

def boolean_blocks_indices(bool_array):
    # Find the indices where the values change from False to True or True to False
    changes = np.where(np.diff(bool_array.astype(int)) != 0)[0] + 1

    # If the first element is True, insert 0 at the beginning
    if bool_array[0]:
        changes = np.insert(changes, 0, 0)

    # If the last element is True, append the array length
    if bool_array[-1]:
        changes = np.append(changes, len(bool_array) + 1)

    # Reshape the indices into pairs
    indices_pairs = changes.reshape(-1, 2)

    # Get the first and last index of each pair
    first_indices = indices_pairs[:, 0]
    last_indices = indices_pairs[:, 1] - 1

    return first_indices, last_indices


def generate_array(row):
    return np.arange(row['activity_start_min'], row['activity_end_min'] + 1)

def combine_arrays(*arrays):
    # Use itertools.product to get all combinations of elements
    combinations = list(product(*arrays))

    # Convert the combinations to a NumPy array
    combined_array = np.array(combinations)

    return combined_array

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def downscale_time(df, new_time):
    # new_time: one time_step in the downscaled df corresponds to new_time*time_step in old df
    # Assuming df has coloumn t 
    df.t = np.floor(df.t / new_time).astype('int')
    return df.drop_duplicates(subset=['i', 'j', 't']).reset_index(drop=True)

def downscale_time_contacts(df, new_time):
    # Similar to downscale time but for contact dataframe
    # new_time: one time_step in the downscaled df corresponds to new_time*time_step in old df
    # Assuming df has coloumn t
    df.start_of_contact = np.floor(df.start_of_contact / new_time).astype('int')
    df.end_of_contact = np.ceil(df.end_of_contact / new_time).astype('int')
    return df.drop_duplicates(subset=['p_A', 'p_B', 'start_of_contact', 'end_of_contact']).reset_index(drop=True)
