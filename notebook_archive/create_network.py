# Used for loading data, creating graphs and collecting them 

import networkx as nx
import pandas as pd
import numpy as np 
import mat73
import glob 
import re
import scipy.io as sio
from networkx.generators.community import LFR_benchmark_graph


# Graph creation
def create_french_school_graph():
    data_dir = "./data/primaryschool.csv"
    df = pd.read_csv(data_dir, sep='\t', header=None)
    df = df.rename({0: 'timestamp', 1: 'id1', 2 : 'id2', 3 : 'class1', 4 : 'class2'}, axis='columns')
    G_raw = nx.Graph()
    all_nodes = set(list(set(list(df.id1))) + list(set(list(df.id2))))
    # all_classes = set(list(set(list(df.id1))) + list(set(list(df.id2))))
    for node in all_nodes:
        G_raw.add_node(node)

    for index, row in df.iterrows():
        if G_raw.has_edge(row["id1"], row["id2"]):
            G_raw[row["id1"]][row["id2"]]["encounters"] += 0.01
        else:
            G_raw.add_edge(row["id1"], row["id2"], encounters=1)

    encounters = nx.get_edge_attributes(G_raw, "encounters")
    return G_raw


def create_lfr_graph(
    n=250, tau1=3, tau2=1.5, mu=0.1):
    G = LFR_benchmark_graph(
        n, tau1, tau2, mu, average_degree=5, min_community=20, seed=10)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def create_erdos_renyi_graph(num_nodes=241, edge_proba=0.1):
    G = nx.erdos_renyi_graph(n=num_nodes, p=edge_proba, seed=10, directed=False)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def load_infectious_graph():
    df = pd.read_csv("./data/Infectious.txt", header=None, delim_whitespace=True)
    G = nx.Graph()
    for index, row in df.iterrows():
        G.add_node(row[0])
        G.add_node(row[1])
        G.add_edge(row[0], row[1])
    return G


def load_contiguous_graph():
    G = nx.read_edgelist("./data/out.contiguous-usa")
    return G


def load_euroroad_graph():
    G = nx.read_edgelist("./data/road-euroroad.edges")
    return G


def load_hamster_graph():
    G = nx.read_edgelist("./data/petster-hamster/out.petster-hamster")
    return G


def load_citeseer_graph():
    df = pd.read_csv("./data/citeseer.edges", header=None,  delimiter=",")
    G = nx.Graph()
    for index, row in df.iterrows():
        G.add_node(row[0])
        G.add_node(row[1])
        G.add_edge(row[0], row[1])
    return G


def load_celegans_graph():
    G = nx.read_edgelist("./data/C-elegans-frontal.txt")
    return G


def load_dolphin_graph():
    a = sio.mmread("./data/soc-dolphins.mtx")
    G = nx.from_scipy_sparse_array(a)
    return G


def load_sfhh_graph():
    df = pd.read_csv("./data/Infectious.txt", header=None, delim_whitespace=True)
    G = nx.Graph()
    for index, row in df.iterrows():
        G.add_node(row[0])
        G.add_node(row[1])
        G.add_edge(row[0], row[1])
    return G


def create_facebook_graph():
    path = "./data/09_Facebook.mat"
    try:
        data_dict = mat73.loadmat(path)
    except:
        data_dict = sio.loadmat(path)
    G = nx.from_scipy_sparse_array(data_dict["facebook4039"])
    return G


def create_email_graph():
    path = "./data/05_Email.mat"
    try:
        data_dict = mat73.loadmat(path)
    except:
        data_dict = sio.loadmat(path)
    G = nx.from_numpy_array(data_dict["Email"])
    return G


def create_network_dat_dataset():
    G_in = load_infectious_graph()
    G_co = load_contiguous_graph()
    G_ce = load_celegans_graph()
    G_do = load_dolphin_graph()
    G_sfhh = load_sfhh_graph()

    return [G_in, G_co, G_ce, G_do, G_sfhh], ["infectious", "contiguous", "celegans", "dolphin", "sfhh"] # 


def create_network14_dataset(exclude=[], include=[]):
    path = "./data/"
    all_graphs, names = create_network_dat_dataset()
    all_graphs.append(create_french_school_graph()) # add french school graph
    names.append('french_school')

    for i in glob.glob(path + '*.mat'):
        try:
            data_dict = mat73.loadmat(i)
        except:
            data_dict = sio.loadmat(i)
        name = re.search('data/(.*).mat', i).group(1)
        name = name[3:]

        if name in exclude:
            continue
        
        if include:
            if not name in include:
                continue

        if name == "NS":
            G = nx.from_numpy_array(data_dict["ns379"])
        elif name == "EEC":
            G = nx.from_scipy_sparse_array(data_dict["email_Eu_core"])
        elif name == "PG":
            G = nx.from_scipy_sparse_array(data_dict["p2p_Gnutella08"])
        elif name == "Enron":
            G = nx.from_scipy_sparse_array(data_dict["enron"])
        elif name == "PB":
            G = nx.from_numpy_array(data_dict["polblogs"])
        elif name == "Facebook":
            G = nx.from_scipy_sparse_array(data_dict["facebook4039"])
        elif name == "WV":
            G = nx.from_scipy_sparse_array(data_dict["Wiki_Vote"])
        elif name == "Sex":
            G = nx.from_scipy_sparse_array(data_dict["sex"])
        elif name == "USAir":
            G = nx.from_numpy_array(data_dict["USAir332"])
        elif name == "Power":
            G = nx.from_numpy_array(data_dict["Power"])
        elif name == "Router":
            G = nx.from_numpy_array(data_dict["Router5022"])
        else:
            G = nx.from_numpy_array(data_dict[name])

        all_graphs.append(G)
        names.append(name)
    
    return all_graphs, names


if __name__ == '__main__':
    pass

