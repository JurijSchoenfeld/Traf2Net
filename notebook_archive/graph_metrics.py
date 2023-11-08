# Investigate network attributes and finde correlations between them

import networkx as nx
from networkx.algorithms import approximation
import notebook_archive.create_network as cn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import time
import signal

class TimeoutException(Exception):   # Custom exception class
    pass


def break_after(seconds=2):
    def timeout_handler(signum, frame):   # Custom signal handler
        raise TimeoutException
    def function(function):
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                res = function(*args, **kwargs)
                signal.alarm(0)      # Clear alarm
                return res
            except TimeoutException:
                print(u'Oops, timeout: %s sec reached.' % seconds, function.__name__, args, kwargs)
            return None, None
        return wrapper
    return function

@break_after(60*10)
def non_randomness_timeout(G):
    return nx.non_randomness(G)


def collect_graph_attributes(G):
    '''Gather topological properties of graphs created in the create netowk module.'''
    degrees = np.array(nx.degree(G))[:, 1]
    max_degree = degrees.max()
    min_degree = degrees.min()
    avg_degree = np.mean(degrees)

    n_nodes = degrees.shape[0]
    n_edges = G.number_of_edges()
    density = nx.density(G)
    
    print('assortativity')
    assortativity = nx.degree_pearson_correlation_coefficient(G)
    print('clustering')
    avg_clustering = nx.average_clustering(G)
    print('diameter')
    diameter = approximation.diameter(G) # nx.diameter(G)
    print('transitivity')
    transitivity = nx.transitivity(G)
    
    # print('randomness')
    # nr, nr_rd = non_randomness_timeout(G)

    print('s metric')
    s_metric = nx.s_metric(G, False)


    attributes = [max_degree, min_degree, avg_degree, n_nodes, n_edges, density, assortativity, \
                  avg_clustering, transitivity, diameter, s_metric]
    
    attribute_names = ['max_degree', 'min_degree', 'avg_degree', 'n_nodes', 'n_edges', 'density', \
                       'assortativity', 'avg_clustering', 'transitivity', 'approx diameter', \
                        's_metric']
    
    return attributes, attribute_names


def structure_graphs():
    '''Structure the attributes gathered in collect_graph_attributes. End result should be a Pandas data frame df_G.'''
    Gs, names = cn.create_network14_dataset(exclude=[])
    del Gs[1:3] # remove 'contagiuous' and 'celegans' as the y throw weird numpy errors
    del names[1:3] # remove 'contagiuous' and 'celegans' as the y throw weird numpy errors
    df = []
    print(names)

    for G, name in zip(Gs, names):
        print(name)
        attributes, attribute_names = collect_graph_attributes(G)
        df.append(attributes)
    
    df_G = pd.DataFrame(np.array(df), columns=attribute_names)
    df_G['name'] = names

    df_G.to_pickle('./pickles/attributes.pkl')

    return df_G


def correlations_cluster_core_nndegree():
    '''Compute correlations between clustering coefficient and nearest neighbour degree/core number.
       Export results as Pandas Dataframe (pickle).'''
    all_Graphs, names = cn.create_network14_dataset(exclude=[])
    df = []

    for G, name in zip(all_Graphs, names):
        print(name)
        core_number = list(nx.core_number(G).values())
        nn_degree = list(nx.average_neighbor_degree(G).values())
        clustering_coef = list(nx.clustering(G).values())

        df.append([*pearsonr(clustering_coef, core_number), *pearsonr(clustering_coef, nn_degree), name])
    
    df_correlations = pd.DataFrame(np.array(df), columns=['$r_{core number, c}$', '$p_{core number, c}$', '$r_{k_{nn}, c}$', '$p_{k_{nn}, c}$', 'name'])
    df_correlations.to_pickle('./pickles/correlations_clust_nndegree_corenumber.pkl')

    return df_correlations
        



if __name__ == '__main__':
    # correlations_cluster_core_nndegree()
    structure_graphs()

    pass
