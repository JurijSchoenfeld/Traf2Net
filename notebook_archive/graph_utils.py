import pandas as pd
import networkx as nx
from tqdm import tqdm
import random
import EoN
import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
import community as community_louvain
from collections import Counter
from networkx.generators.community import LFR_benchmark_graph
import mat73
import glob 
import re
import scipy.io as sio
from statistics import median, mean
import math
from scipy.io import mmread

plot_dir = "/localdata2/dial_mo/risk_analysis/privacy_preserving_risk_of_infection/plots/"
graph_dir = "/localdata2/dial_mo/risk_analysis/privacy_preserving_risk_of_infection/graphs/"


# Graph creation
def create_french_school_graph():
    data_dir = "/localdata2/dial_mo/risk_analysis/privacy_preserving_risk_of_infection/data/primaryschool.csv"
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
    df = pd.read_csv("/localdata2/dial_mo/risk_analysis/privacy_preserving_risk_of_infection/data/Infectious.txt", header=None, delim_whitespace=True)
    G = nx.Graph()
    for index, row in df.iterrows():
        G.add_node(row[0])
        G.add_node(row[1])
        G.add_edge(row[0], row[1])
    return G

def load_contiguous_graph():
    G = nx.read_edgelist("/localdata2/dial_mo/risk_analysis/privacy_preserving_risk_of_infection/data/out.contiguous-usa")
    return G

def load_euroroad_graph():
    G = nx.read_edgelist("/localdata2/dial_mo/risk_analysis/privacy_preserving_risk_of_infection/data/road-euroroad.edges")
    return G

def load_hamster_graph():
    G = nx.read_edgelist("/localdata2/dial_mo/risk_analysis/privacy_preserving_risk_of_infection/data/petster-hamster/out.petster-hamster")
    return G

def load_citeseer_graph():
    df = pd.read_csv("/localdata2/dial_mo/risk_analysis/privacy_preserving_risk_of_infection/data/citeseer.edges", header=None,  delimiter=",")
    G = nx.Graph()
    for index, row in df.iterrows():
        G.add_node(row[0])
        G.add_node(row[1])
        G.add_edge(row[0], row[1])
    return G

def load_celegans_graph():
    G = nx.read_edgelist("/localdata2/dial_mo/risk_analysis/privacy_preserving_risk_of_infection/data/C-elegans-frontal.txt")
    return G

def load_dolphin_graph():
    a = mmread("/localdata2/dial_mo/risk_analysis/privacy_preserving_risk_of_infection/data/soc-dolphins.mtx")
    G = nx.from_scipy_sparse_matrix(a)
    return G

def load_sfhh_graph():
    df = pd.read_csv("/localdata2/dial_mo/risk_analysis/privacy_preserving_risk_of_infection/data/Infectious.txt", header=None, delim_whitespace=True)
    G = nx.Graph()
    for index, row in df.iterrows():
        G.add_node(row[0])
        G.add_node(row[1])
        G.add_edge(row[0], row[1])
    return G

def create_network_dat_dataset():
    G_in = load_infectious_graph()
    G_co = load_contiguous_graph()
    G_ce = load_celegans_graph()
    G_do = load_dolphin_graph()
    G_sfhh = load_sfhh_graph()

    return [G_in, G_co, G_ce, G_do, G_sfhh], ["infectious", "contiguous", "celegans", "dolphin", "sfhh"] # 

def create_network14_dataset(exclude=[], include=[]):
    path = "/localdata2/dial_mo/risk_analysis/privacy_preserving_risk_of_infection/data/"
    all_graphs = []
    names = []
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
            G = nx.from_numpy_matrix(data_dict["ns379"])
        elif name == "EEC":
            G = nx.from_scipy_sparse_matrix(data_dict["email_Eu_core"])
        elif name == "PG":
            G = nx.from_scipy_sparse_matrix(data_dict["p2p_Gnutella08"])
        elif name == "Enron":
            G = nx.from_scipy_sparse_matrix(data_dict["enron"])
        elif name == "PB":
            G = nx.from_numpy_matrix(data_dict["polblogs"])
        elif name == "Facebook":
            G = nx.from_scipy_sparse_matrix(data_dict["facebook4039"])
        elif name == "WV":
            G = nx.from_scipy_sparse_matrix(data_dict["Wiki_Vote"])
        elif name == "Sex":
            G = nx.from_scipy_sparse_matrix(data_dict["sex"])
        elif name == "USAir":
            G = nx.from_numpy_matrix(data_dict["USAir332"])
        elif name == "Power":
            G = nx.from_numpy_matrix(data_dict["Power"])
        elif name == "Router":
            G = nx.from_numpy_matrix(data_dict["Router5022"])
        else:
            G = nx.from_numpy_matrix(data_dict[name])
        all_graphs.append(G)
        names.append(name)
    return all_graphs, names

def create_facebook_graph():
    path = "/localdata2/dial_mo/risk_analysis/privacy_preserving_risk_of_infection/data/09_Facebook.mat"
    try:
        data_dict = mat73.loadmat(path)
    except:
        data_dict = sio.loadmat(path)
    G = nx.from_scipy_sparse_matrix(data_dict["facebook4039"])
    return G

def create_email_graph():
    path = "/localdata2/dial_mo/risk_analysis/privacy_preserving_risk_of_infection/data/05_Email.mat"
    try:
        data_dict = mat73.loadmat(path)
    except:
        data_dict = sio.loadmat(path)
    G = nx.from_numpy_matrix(data_dict["Email"])
    return G

def get_num_infections_per_node(nodes, sim_run, node_dict, init_node):
    count_infections = 0
    for node in nodes:
        if init_node == node:
            continue
        node_info = sim_run.node_history(node)
        if "I" in node_info[1]:
            if "I" in node_info[1][0]:
                continue
            else:
                node_dict[node] += 1
                count_infections += 1
    return node_dict

def get_num_infected_nodes(nodes, sim_run, init_node):
    count_infections = 0
    for node in nodes:
        if init_node == node:
            continue
        node_info = sim_run.node_history(node)
        if "R" in node_info[1]:
            if "R" in node_info[1][0]:
                continue
            else:
                count_infections += 1
    return count_infections

def _process_attributes(G_cut):
    degrees = [val for (node, val) in G_cut.degree()]
    degree_dict = dict(zip(list(G_cut.nodes()), degrees)) 

    # avg_shortest_path = nx.average_shortest_path_length(G_cut)
    clustering_coefficient = nx.clustering(G_cut)
    try:
        eigenvec_centrality_dic = nx.eigenvector_centrality(G_cut)
    except:
        eigenvec_centrality_dic = dict(zip(list(G_cut.nodes()), [0] * len(list(G_cut.nodes()))))
    betweeness_centrality_dic = nx.betweenness_centrality(G_cut)
    closeness_centrality_dic = nx.closeness_centrality(G_cut)
    core_num_dict = nx.core_number(G_cut) 
    path_length_dict = path_lengths(G_cut)
    core_num_alternative_dict, core_num_iteration_dict = k_shell_alternative(G_cut)
    i_kshell_dict = i_kshell(G_cut, core_num_dict, core_num_iteration_dict)
    all_paths = []
    for key, value in path_length_dict.items():
        all_paths.extend(value.values())
    avg_shortest_path = mean(all_paths)

    density_centrality_dict = density_centrality(G_cut, path_length_dict, degree_dict)

    grav_centrality_dict = gravity_centrality(G_cut, path_length_dict, core_num_dict)
    grav_centrality_plus_dict = gravity_centrality_plus(G_cut, grav_centrality_dict)

    improved_grav_centrality_dict = improved_gravity_centrality(G_cut, path_length_dict, core_num_dict, degree_dict)
    improved_grav_centrality_plus_dict = improved_gravity_centrality_plus(G_cut, improved_grav_centrality_dict)
    improved_gravity_centrality2_dict = improved_gravity_centrality2(G_cut, path_length_dict, degree_dict, i_kshell_dict)
    improved_gravity_centrality2_plus_dict = improved_gravity_centrality2_plus(G_cut, improved_gravity_centrality2_dict)
    local_gravity_centrality_dict = local_gravity_centrality(G_cut, path_length_dict, core_num_dict, degree_dict, avg_shortest_path)
    mcgm_dict = mcgm(G_cut, path_length_dict, core_num_dict, degree_dict, eigenvec_centrality_dic, avg_shortest_path)
    

    gli_method_dict = gli_method(G_cut, i_kshell_dict, degree_dict, path_length_dict)
    clustered_local_degree_dict = clustered_local_degree(G_cut, path_length_dict, clustering_coefficient, degree_dict)
    # constraint_dict = nx.constraint(G_cut)
    
    alters_degrees = [val for (node, val) in G_cut.degree()]
    alters_degree_dict = dict(zip(list(G_cut.nodes()), alters_degrees))
    ninl_dict = ninl(G_cut, degree_dict, path_length_dict, avg_shortest_path)
    
    density = nx.density(G_cut)
    diameter = nx.diameter(G_cut)

    # new_measure1_dict = new_measure1(G_cut, path_length_dict, degree_dict, i_kshell_dict, avg_shortest_path)
    # new_measure2_dict = new_measure2(G_cut, path_length_dict, degree_dict, i_kshell_dict, avg_shortest_path)
    # new_measure3_dict = new_measure3(G_cut, path_length_dict, degree_dict, i_kshell_dict, avg_shortest_path)
    # new_measure4_dict = new_measure4(G_cut, path_length_dict, degree_dict, i_kshell_dict, avg_shortest_path)
    # new_measure5_dict = new_measure5(G_cut, path_length_dict, degree_dict, i_kshell_dict, avg_shortest_path)

    ks_if_dict = ks_if(G_cut, core_num_dict, core_num_iteration_dict, degree_dict)
    mcde_dict, mcdwe_dict = mcde(G_cut, core_num_dict, degree_dict,)
    erm_dict = erm(G_cut, degree_dict,)
    new_gli_method_dict = new_gli_method(G_cut, core_num_dict, degree_dict,)
    h_index_dict = h_index(G_cut)
    local_h_index_dict = local_h_index(G_cut, h_index_dict)
    dsr_dict, edsr_dict = dsr(G_cut, degree_dict, core_num_dict, core_num_iteration_dict)
    LS_dict = LS(G_cut, degree_dict, core_num_dict)
    ECRM_dict = ECRM(G_cut, degree_dict, core_num_iteration_dict)

    return degree_dict, clustering_coefficient,\
            eigenvec_centrality_dic, \
            betweeness_centrality_dic, closeness_centrality_dic, \
            core_num_dict, path_length_dict, \
            core_num_iteration_dict, core_num_alternative_dict, \
            i_kshell_dict, avg_shortest_path, \
            density_centrality_dict, grav_centrality_dict, \
            grav_centrality_plus_dict, improved_grav_centrality_dict, \
            improved_grav_centrality_plus_dict, improved_gravity_centrality2_dict, \
            improved_gravity_centrality2_plus_dict, local_gravity_centrality_dict, \
            mcgm_dict, gli_method_dict, \
            clustered_local_degree_dict, alters_degree_dict, \
            ninl_dict, density, \
            diameter, \
            ks_if_dict, mcde_dict, mcdwe_dict, \
            erm_dict, new_gli_method_dict, h_index_dict, \
            local_h_index_dict, dsr_dict, edsr_dict, \
            LS_dict, ECRM_dict

def _process_specific_attributes(G_cut):
    degrees = [val for (node, val) in G_cut.degree()]
    degree_dict = dict(zip(list(G_cut.nodes()), degrees)) 

    core_num_dict = nx.core_number(G_cut) 
    core_num_alternative_dict, core_num_iteration_dict = k_shell_alternative(G_cut)

    ks_if_dict = ks_if(G_cut, core_num_dict, core_num_iteration_dict, degree_dict)
    mcde_dict, mcdwe_dict = mcde(G_cut, core_num_dict, degree_dict,)
    erm_dict = erm(G_cut, degree_dict,)
    new_gli_method_dict = new_gli_method(G_cut, core_num_dict, degree_dict,)
    h_index_dict = h_index(G_cut)
    local_h_index_dict = local_h_index(G_cut, h_index_dict)
    dsr_dict, edsr_dict = dsr(G_cut, degree_dict, core_num_dict, core_num_iteration_dict)
    LS_dict = LS(G_cut, degree_dict, core_num_dict)
    ECRM_dict = ECRM(G_cut, degree_dict, core_num_iteration_dict, core_num_dict)

    return degree_dict, \
            core_num_dict, \
            core_num_iteration_dict, core_num_alternative_dict, \
            ks_if_dict, mcde_dict, mcdwe_dict, \
            erm_dict, new_gli_method_dict, h_index_dict, \
            local_h_index_dict, dsr_dict, edsr_dict, \
            LS_dict, ECRM_dict

def _set_attributes(attrs, K_name, n, G_cut, G,
                    degree_dict, clustering_coefficient,\
                    eigenvec_centrality_dic, \
                    betweeness_centrality_dic, closeness_centrality_dic, \
                    core_num_dict, path_length_dict, \
                    core_num_iteration_dict, core_num_alternative_dict, \
                    i_kshell_dict, avg_shortest_path, \
                    density_centrality_dict, grav_centrality_dict, \
                    grav_centrality_plus_dict, improved_grav_centrality_dict, \
                    improved_grav_centrality_plus_dict, improved_gravity_centrality2_dict, \
                    improved_gravity_centrality2_plus_dict, local_gravity_centrality_dict, \
                    mcgm_dict, gli_method_dict, \
                    clustered_local_degree_dict, alters_degree_dict, \
                    ninl_dict, density, \
                    diameter, \
                    # new_measure1_dict, new_measure2_dict, new_measure3_dict, \
                    # new_measure4_dict, new_measure5_dict, \
                    ks_if_dict, mcde_dict, mcdwe_dict, \
                    erm_dict, new_gli_method_dict, h_index_dict, \
                    local_h_index_dict, dsr_dict, edsr_dict, \
                    LS_dict, ECRM_dict):

    

    K_name = str(K_name)
    attrs["i_kshell" + "_" + K_name] = i_kshell_dict[n]
    attrs["density_centrality" + "_" + K_name] = density_centrality_dict[n]
    attrs["gli_method" + "_" + K_name] = gli_method_dict[n]
    attrs["clustered_local_degree" + "_" + K_name] = clustered_local_degree_dict[n]
    attrs["ninl" + "_" + K_name] = ninl_dict[n]
            
    # Node degree
    attrs["node_degree" + "_" + K_name] = degree_dict[n]
    attrs["alters_avg_node_degree" + "_" + K_name] = mean(alters_degree_dict.values())
    # Egonet
    attrs["egonet_edge_num" + "_" + K_name] = G_cut.number_of_edges()
    # Triangles
    attrs["triangles" + "_" + K_name] = nx.triangles(G_cut, n)
    # Deepwalk embedding
    random_walk_length = get_random_walk(G_cut, n, 10)
    attrs["deepwalk_vec_length" + "_" + K_name] = len(random_walk_length)
    attrs["deepwalk_vec_avg" + "_" + K_name] = int(sum([int(x) for x in random_walk_length]) / len(random_walk_length))
    # Local cc
    attrs["local_cc" + "_" + K_name] = local_cc(G_cut, G)
    # eigenvector_centrality
    attrs["eigenvector_centrality" + "_" + K_name] = eigenvec_centrality_dic[n]
    # betweenness_centrality
    attrs["betweenness_centrality" + "_" + K_name] = betweeness_centrality_dic[n]
    # closeness_centrality
    attrs["closeness_centrality" + "_" + K_name] = closeness_centrality_dic[n]
    # k_shell_dict
    attrs["k_core" + "_" + K_name] = core_num_dict[n]
    # gravity_centrality
    attrs["gravity_centrality" + "_" + K_name] = grav_centrality_dict[n]
    attrs["grav_centrality_plus" + "_" + K_name] = grav_centrality_plus_dict[n]
    attrs["improved_gravity_centrality" + "_" + K_name] = improved_grav_centrality_dict[n]
    attrs["improved_gravity_centrality_plus" + "_" + K_name] = improved_grav_centrality_plus_dict[n]
    attrs["improved_gravity_centrality2" + "_" + K_name] = improved_gravity_centrality2_dict[n]
    attrs["improved_gravity_centrality2_plus" + "_" + K_name] = improved_gravity_centrality2_plus_dict[n]
    attrs["local_gravity_centrality" + "_" + K_name] = local_gravity_centrality_dict[n]
    attrs["mcgm" + "_" + K_name] = mcgm_dict[n]
    # structural holes/constraint
    # attrs['constraint' + "_" + K_name] = constraint_dict[n]
    # density
    attrs["density" + "_" + K_name] = density
    attrs["diameter" + "_" + K_name] = diameter
    attrs["avg_shortest_path" + "_" + K_name] = avg_shortest_path

    # attrs["new_measure1" + "_" + K_name] = new_measure1_dict[n]
    # attrs["new_measure2" + "_" + K_name] = new_measure2_dict[n]
    # attrs["new_measure3" + "_" + K_name] = new_measure3_dict[n]
    # attrs["new_measure4" + "_" + K_name] = new_measure4_dict[n]
    # attrs["new_measure5" + "_" + K_name] = new_measure5_dict[n]

    attrs["ks_if" + "_" + K_name] = ks_if_dict[n]
    attrs["mcde" + "_" + K_name] = mcde_dict[n]
    attrs["mcdwe" + "_" + K_name] = mcdwe_dict[n]
    attrs["erm" + "_" + K_name] = erm_dict[n]
    attrs["new_gli_method" + "_" + K_name] = new_gli_method_dict[n]
    attrs["h_index" + "_" + K_name] = h_index_dict[n]
    attrs["local_h_index" + "_" + K_name] = local_h_index_dict[n]
    attrs["dsr" + "_" + K_name] = dsr_dict[n]
    attrs["edsr" + "_" + K_name] = edsr_dict[n]
    attrs["LS" + "_" + K_name] = LS_dict[n]
    attrs["ECRM" + "_" + K_name] = ECRM_dict[n]

    return attrs

def _set_specific_attributes(attrs, K_name, n, G_cut, G,
                            degree_dict, \
                            core_num_dict, \
                            core_num_iteration_dict, core_num_alternative_dict, \
                            ks_if_dict, mcde_dict, mcdwe_dict, \
                            erm_dict, new_gli_method_dict, h_index_dict, \
                            local_h_index_dict, dsr_dict, edsr_dict, \
                            LS_dict, ECRM_dict):

    K_name = str(K_name)

    attrs["ks_if" + "_" + K_name] = ks_if_dict[n]
    attrs["mcde" + "_" + K_name] = mcde_dict[n]
    attrs["mcdwe" + "_" + K_name] = mcdwe_dict[n]
    attrs["erm" + "_" + K_name] = erm_dict[n]
    attrs["new_gli_method" + "_" + K_name] = new_gli_method_dict[n]
    attrs["h_index" + "_" + K_name] = h_index_dict[n]
    attrs["local_h_index" + "_" + K_name] = local_h_index_dict[n]
    attrs["dsr" + "_" + K_name] = dsr_dict[n]
    attrs["edsr" + "_" + K_name] = edsr_dict[n]
    attrs["LS" + "_" + K_name] = LS_dict[n]
    attrs["ECRM" + "_" + K_name] = ECRM_dict[n]

    return attrs

def create_node_attributes_full_graph(G):
    K_name = "full_graph"
    print("K_name: " + str(K_name))

    degree_dict, clustering_coefficient,\
    eigenvec_centrality_dic, \
    betweeness_centrality_dic, closeness_centrality_dic, \
    core_num_dict, path_length_dict, \
    core_num_iteration_dict, core_num_alternative_dict, \
    i_kshell_dict, avg_shortest_path, \
    density_centrality_dict, grav_centrality_dict, \
    grav_centrality_plus_dict, improved_grav_centrality_dict, \
    improved_grav_centrality_plus_dict, improved_gravity_centrality2_dict, \
    improved_gravity_centrality2_plus_dict, local_gravity_centrality_dict, \
    mcgm_dict, gli_method_dict, \
    clustered_local_degree_dict, alters_degree_dict, \
    ninl_dict, density, \
    diameter, \
    ks_if_dict, mcde_dict, mcdwe_dict, \
    erm_dict, new_gli_method_dict, h_index_dict, \
    local_h_index_dict, dsr_dict, edsr_dict, \
    LS_dict, ECRM_dict = _process_attributes(G)
    
    for n, attrs in tqdm(G.nodes(data=True), total=len(list(G.nodes(data=True))), desc="full graph.."):
        attrs = _set_attributes(attrs, K_name, n, G, G,
                degree_dict, clustering_coefficient,\
                eigenvec_centrality_dic, \
                betweeness_centrality_dic, closeness_centrality_dic, \
                core_num_dict, path_length_dict, \
                core_num_iteration_dict, core_num_alternative_dict, \
                i_kshell_dict, avg_shortest_path, \
                density_centrality_dict, grav_centrality_dict, \
                grav_centrality_plus_dict, improved_grav_centrality_dict, \
                improved_grav_centrality_plus_dict, improved_gravity_centrality2_dict, \
                improved_gravity_centrality2_plus_dict, local_gravity_centrality_dict, \
                mcgm_dict, gli_method_dict, \
                clustered_local_degree_dict, alters_degree_dict, \
                ninl_dict, density, \
                diameter, \
                # new_measure1_dict, new_measure2_dict, new_measure3_dict, \
                # new_measure4_dict, new_measure5_dict, \
                ks_if_dict, mcde_dict, mcdwe_dict, \
                erm_dict, new_gli_method_dict, h_index_dict, \
                local_h_index_dict, dsr_dict, edsr_dict, \
                LS_dict, ECRM_dict)
    return G

def create_specific_node_attributes_full_graph(G):
    K_name = "full_graph"
    print("K_name: " + str(K_name))

    degree_dict, \
    core_num_dict, \
    core_num_iteration_dict, core_num_alternative_dict, \
    ks_if_dict, mcde_dict, mcdwe_dict, \
    erm_dict, new_gli_method_dict, h_index_dict, \
    local_h_index_dict, dsr_dict, edsr_dict, \
    LS_dict, ECRM_dict = _process_specific_attributes(G)

    for n, attrs in tqdm(G.nodes(data=True), total=len(list(G.nodes(data=True))), desc="full graph.."):
        attrs = _set_specific_attributes(attrs, K_name, n, G, G, \
                    degree_dict, \
                    core_num_dict, \
                    core_num_iteration_dict, core_num_alternative_dict, \
                    ks_if_dict, mcde_dict, mcdwe_dict, \
                    erm_dict, new_gli_method_dict, h_index_dict, \
                    local_h_index_dict, dsr_dict, edsr_dict, \
                    LS_dict, ECRM_dict)
    return G

def create_node_attributes(G, K_range=0, single_K=None):
    K_names = []
    for k in range(K_range+1):
        K_names.append(str(k) + "_hop")

    # degrees = [val for (node, val) in G.degree()]
    # degree_dict = dict(zip(list(G.nodes()), degrees))
    
    hop_list = list(range(K_range+1))
    # hop_list.append(diameter_G)
    if single_K:
        K_names = [str(single_K) + "_hop"]
        hop_list = [single_K]
    
    for K, K_name in tqdm(zip(hop_list, K_names), total=K_range, desc="Create attributes"):
        count = 0
        print("K_name: " + str(K_name))
        print("\n")
        for n, attrs in tqdm(G.nodes(data=True), total=len(list(G.nodes())), desc=""):
            # ego_G = nx.ego_graph(G, n)
            # print(K_name)
            if K > 0:
                cut = list(nx.single_source_shortest_path_length(G, n, cutoff=K).keys())
                G_cut = nx.induced_subgraph(G, cut)
            else:
                cut_nodes = list(G.neighbors(n))
                cut_edges = list(G.edges(n))
                G_cut = nx.Graph()
                G_cut.add_nodes_from(cut_nodes)
                G_cut.add_edges_from(cut_edges)
            
            degree_dict, clustering_coefficient,\
            eigenvec_centrality_dic, \
            betweeness_centrality_dic, closeness_centrality_dic, \
            core_num_dict, path_length_dict, \
            core_num_iteration_dict, core_num_alternative_dict, \
            i_kshell_dict, avg_shortest_path, \
            density_centrality_dict, grav_centrality_dict, \
            grav_centrality_plus_dict, improved_grav_centrality_dict, \
            improved_grav_centrality_plus_dict, improved_gravity_centrality2_dict, \
            improved_gravity_centrality2_plus_dict, local_gravity_centrality_dict, \
            mcgm_dict, gli_method_dict, \
            clustered_local_degree_dict, alters_degree_dict, \
            ninl_dict, density, \
            diameter, \
            ks_if_dict, mcde_dict, mcdwe_dict, \
            erm_dict, new_gli_method_dict, h_index_dict, \
            local_h_index_dict, dsr_dict, edsr_dict, \
            LS_dict, ECRM_dict = _process_attributes(G_cut)
            
            attrs = _set_attributes(attrs, K_name, n, G_cut, G,
                    degree_dict, clustering_coefficient,\
                    eigenvec_centrality_dic, \
                    betweeness_centrality_dic, closeness_centrality_dic, \
                    core_num_dict, path_length_dict, \
                    core_num_iteration_dict, core_num_alternative_dict, \
                    i_kshell_dict, avg_shortest_path, \
                    density_centrality_dict, grav_centrality_dict, \
                    grav_centrality_plus_dict, improved_grav_centrality_dict, \
                    improved_grav_centrality_plus_dict, improved_gravity_centrality2_dict, \
                    improved_gravity_centrality2_plus_dict, local_gravity_centrality_dict, \
                    mcgm_dict, gli_method_dict, \
                    clustered_local_degree_dict, alters_degree_dict, \
                    ninl_dict, density, \
                    diameter, \
                    # new_measure1_dict, new_measure2_dict, new_measure3_dict, \
                    # new_measure4_dict, new_measure5_dict, \
                    ks_if_dict, mcde_dict, mcdwe_dict, \
                    erm_dict, new_gli_method_dict, h_index_dict, \
                    local_h_index_dict, dsr_dict, edsr_dict, \
                    LS_dict, ECRM_dict)
                
            count += 1

    if not single_K:
        G = create_node_attributes_full_graph(G)
    return G
        # neighbors = G.edges(n)
        # current_weights = []

def create_specific_node_attributes(G, K_range=0, single_K=None):
    K_names = []
    for k in range(K_range+1):
        K_names.append(str(k) + "_hop")

    hop_list = list(range(K_range+1))
    if single_K:
        K_names = [str(single_K) + "_hop"]
        hop_list = [single_K]
    
    for K, K_name in tqdm(zip(hop_list, K_names), total=K_range, desc="Create attributes"):
        count = 0
        print("K_name: " + str(K_name))
        print("\n")
        for n, attrs in tqdm(G.nodes(data=True), total=len(list(G.nodes())), desc=""):
            # ego_G = nx.ego_graph(G, n)
            # print(K_name)
            if K > 0:
                cut = list(nx.single_source_shortest_path_length(G, n, cutoff=K).keys())
                G_cut = nx.induced_subgraph(G, cut)
            else:
                cut_nodes = list(G.neighbors(n))
                cut_edges = list(G.edges(n))
                G_cut = nx.Graph()
                G_cut.add_nodes_from(cut_nodes)
                G_cut.add_edges_from(cut_edges)
            
            degree_dict, \
            core_num_dict, \
            core_num_iteration_dict, core_num_alternative_dict, \
            ks_if_dict, mcde_dict, mcdwe_dict, \
            erm_dict, new_gli_method_dict, h_index_dict, \
            local_h_index_dict, dsr_dict, edsr_dict, \
            LS_dict, ECRM_dict = _process_specific_attributes(G_cut)
            
            attrs = _set_specific_attributes(attrs, K_name, n , G_cut, G, \
                    degree_dict, \
                    core_num_dict, \
                    core_num_iteration_dict, core_num_alternative_dict, \
                    ks_if_dict, mcde_dict, mcdwe_dict, \
                    erm_dict, new_gli_method_dict, h_index_dict, \
                    local_h_index_dict, dsr_dict, edsr_dict, \
                    LS_dict, ECRM_dict)
                
            count += 1

    if not single_K:
        G = create_specific_node_attributes_full_graph(G)
    return G


def create_ninl_attribute(G, K_range, attr_name="ninl"):
    K_names = []
    for k in range(K_range+1):
        K_names.append(str(k) + "_hop")

    # degrees = [val for (node, val) in G.degree()]
    # degree_dict = dict(zip(list(G.nodes()), degrees))
    
    hop_list = list(range(K_range+1))
    # hop_list.append(diameter_G)


    for K, K_name in tqdm(zip(hop_list, K_names), total=K_range, desc="K-hop neighborhood.."):
        count = 0
        print(K_name)
        print("\n")
        for n, attrs in tqdm(G.nodes(data=True), total=len(list(G.nodes())), desc="Iterate through nodes.."):
            # ego_G = nx.ego_graph(G, n)
            # print(K_name)
            if K > 0:
                cut = list(nx.single_source_shortest_path_length(G, n, cutoff=K).keys())
                G_cut = nx.induced_subgraph(G, cut)
            else:
                cut_nodes = list(G.neighbors(n))
                cut_edges = list(G.edges(n))
                G_cut = nx.Graph()
                G_cut.add_nodes_from(cut_nodes)
                G_cut.add_edges_from(cut_edges)
            
            degrees = [val for (node, val) in G_cut.degree()]
            degree_dict = dict(zip(list(G_cut.nodes()), degrees))
            
            # avg_shortest_path = nx.average_shortest_path_length(G_cut)
            path_length_dict = path_lengths(G_cut)

            all_paths = []
            for key, value in path_length_dict.items():
                all_paths.extend(value.values())
            avg_shortest_path = mean(all_paths)
            ninl_dict = ninl(G_cut, degree_dict, path_length_dict, avg_shortest_path)
            attrs["ninl" + "_" + K_name] = ninl_dict[n]
    

    degrees = [val for (node, val) in G.degree()]
    degree_dict = dict(zip(list(G.nodes()), degrees))
    # avg_shortest_path = nx.average_shortest_path_length(G)
    path_length_dict = path_lengths(G)
    all_paths = []
    for key, value in path_length_dict.items():
        all_paths.extend(value.values())
    avg_shortest_path = mean(all_paths)
    ninl_dict = ninl(G, degree_dict, path_length_dict, avg_shortest_path)
    
    K_name = "full_graph"
    for n, attrs in tqdm(G.nodes(data=True), total=len(list(G.nodes())), desc="Iterate through nodes.."):
        attrs["ninl" + "_" + K_name] = ninl_dict[n]
    return G

def create_new_measure_attr(G, K_range=3, only_full_graph=False, single_K=None):
    K_names = []
    for k in range(K_range+1):
        K_names.append(str(k) + "_hop")

    hop_list = list(range(K_range+1))

    if single_K:
        K_names = [str(single_K) + "_hop"]
        hop_list = [single_K]

    if not only_full_graph:
        for K, K_name in tqdm(zip(hop_list, K_names), total=K_range, desc="Create node attributes..."):
            count = 0
            print("\n")
            print(K_name)
            print("\n")
            for n, attrs in tqdm(G.nodes(data=True), total=len(list(G.nodes())), desc="Iterate through nodes.."):
                # ego_G = nx.ego_graph(G, n)
                # print(K_name)
                if K > 0:
                    cut = list(nx.single_source_shortest_path_length(G, n, cutoff=K).keys())
                    G_cut = nx.induced_subgraph(G, cut)
                else:
                    cut_nodes = list(G.neighbors(n))
                    cut_edges = list(G.edges(n))
                    G_cut = nx.Graph()
                    G_cut.add_nodes_from(cut_nodes)
                    G_cut.add_edges_from(cut_edges)
                
                degrees = [val for (node, val) in G_cut.degree()]
                degree_dict = dict(zip(list(G_cut.nodes()), degrees))
                
                # avg_shortest_path = nx.average_shortest_path_length(G_cut)
                path_length_dict = path_lengths(G_cut)

                all_paths = []
                for key, value in path_length_dict.items():
                    all_paths.extend(value.values())
                avg_shortest_path = mean(all_paths)
                
                ninl_dict = ninl(G_cut, degree_dict, path_length_dict, avg_shortest_path)
                attrs["ninl" + "_" + K_name] = ninl_dict[n]

                core_num_dict = nx.core_number(G_cut) 
                core_num_alternative_dict, core_num_iteration_dict = k_shell_alternative(G_cut)
                i_kshell_dict = i_kshell(G_cut, core_num_dict, core_num_iteration_dict)
                measure_dict4 = new_measure4(G_cut, path_length_dict, degree_dict, i_kshell_dict, avg_shortest_path)
                measure_dict5 = new_measure5(G_cut, path_length_dict, degree_dict, i_kshell_dict, avg_shortest_path)
                attrs["new_measure4" + "_" + K_name] = measure_dict4[n]
                attrs["new_measure5" + "_" + K_name] = measure_dict5[n]
        

    degrees = [val for (node, val) in G.degree()]
    degree_dict = dict(zip(list(G.nodes()), degrees))
    # avg_shortest_path = nx.average_shortest_path_length(G)
    path_length_dict = path_lengths(G)
    core_num_dict = nx.core_number(G) 
    core_num_alternative_dict, core_num_iteration_dict = k_shell_alternative(G)
    i_kshell_dict = i_kshell(G, core_num_dict, core_num_iteration_dict)
    all_paths = []
    for key, value in path_length_dict.items():
        all_paths.extend(value.values())
    avg_shortest_path = mean(all_paths)
    measure_dict4 = new_measure4(G, path_length_dict, degree_dict, i_kshell_dict, avg_shortest_path)
    measure_dict5 = new_measure5(G, path_length_dict, degree_dict, i_kshell_dict, avg_shortest_path)
    
    K_name = "full_graph"
    for n, attrs in tqdm(G.nodes(data=True), total=len(list(G.nodes())), desc="Iterate through nodes.."):
        attrs["new_measure4" + "_" + K_name] = measure_dict4[n]
        attrs["new_measure5" + "_" + K_name] = measure_dict5[n]
    return G


def create_node_labels(G, node_dict, influence_node_dict, num_sim_runs, num_classes=10):
    # Create labels
    # node_dict = get_num_infections_per_node(nodes, sim_run)
    rel_values = [float(x) / float(num_sim_runs) * 1000 for x in node_dict.values()]
    # rel_values = [round(float(x) / float(len(simulation_runs)) * 10, 2) for x in node_dict.values()]
    rel_values_raw = [float(x) / float(num_sim_runs) for x in node_dict.values()]
    rel_values_abs = [int(float(x) / float(num_sim_runs) * 1000) for x in node_dict.values()]
    input_start = 0
    input_end = max(rel_values)
    output_start = 0
    output_end = num_classes
    slope = (output_end - output_start) / (input_end - input_start)
    new_rel_values = []
    new_rel_values_abs = []
    for val in rel_values:
        new_rel_values.append(output_start + slope * (val - input_start))
    for val in rel_values_abs:
        new_rel_values_abs.append(int(output_start + slope * (val - input_start)))

    relative_node_dict_mapped = dict(zip(node_dict.keys(), new_rel_values))
    relative_node_dict_raw = dict(zip(node_dict.keys(), rel_values_raw))
    relative_node_dict_abs = dict(zip(node_dict.keys(), rel_values_abs))
    relative_node_dict_abs_mapped = dict(zip(node_dict.keys(), new_rel_values_abs))
    attrs_node_dict = {}
    for n, attrs in G.nodes(data=True):
        attrs_node_dict[n] = {"risk_mapped": relative_node_dict_mapped[n],
                              "risk_raw": relative_node_dict_raw[n],
                              "risk_abs": relative_node_dict_abs[n],
                              "risk_abs_mapped": relative_node_dict_abs_mapped[n],
                              "influence_score": influence_node_dict[n]}
    G_copy = G.copy()
    nx.set_node_attributes(G_copy, attrs_node_dict)
    return G_copy

def get_random_walk(ego_graph, node, walk_length):
    # initialization
    random_walk_length = [node]
    
    #loop over to get the nodes visited in a random walk
    for i in range(walk_length-1):
        # list of neighbors
        neighbors = list(ego_graph.neighbors(node))
        # if the same neighbors are present in ranom_walk_length list, then donot add them as new neighbors
        neighbors = list(set(neighbors) - set(random_walk_length))    
        if len(neighbors) == 0:
            break
        # pick any one neighbor randomly from the neighbors list
        random_neighbor = random.choice(neighbors)
        # append that random_neighbor to the random_walk_length list
        random_walk_length.append(random_neighbor)
        node = random_neighbor
        
    return random_walk_length

def local_cc(ego_graph, cur_G):
    communities = community_louvain.best_partition(ego_graph)
    coms = {}
    for k, v in communities.items():
        if v in coms:
            coms[v].append(k)
        else:
            coms[v] = [k]
    final_local_cc = []
    for com in coms.values():
        subg = nx.induced_subgraph(cur_G, list(com))
        num_edges = len(list(subg.edges()))
        num_nodes = len(list(subg.nodes()))
        loc_cc = (num_edges + 1) * num_nodes
        final_local_cc.append(loc_cc)
    return sum(final_local_cc)

def path_lengths(G):
    path_length_dict = dict(nx.all_pairs_shortest_path_length(G))
    return path_length_dict

def gravity_centrality(G, path_length_dict, core_num_dict):
    # https://arxiv.org/abs/1505.02476
    grav_centrality = []
    for n1, attrs in G.nodes(data=True):
        gc = 0
        for n2, attrs in G.nodes(data=True):
            if n1 in path_length_dict and n2 in path_length_dict[n1]: 
                if path_length_dict[n1][n2] <= 3 and n1 != n2:
                    node_gravity = (core_num_dict[n1] * core_num_dict[n2]) / (path_length_dict[n1][n2] ** 2)
                    gc += node_gravity
                else:
                    continue
            else:
                continue
        grav_centrality.append(gc)
    return dict(zip(list(G.nodes()), grav_centrality))

def gravity_centrality_plus(G, grav_centrality):
    grav_centrality_plus = []
    
    for n1, attrs in G.nodes(data=True):
        gc = 0
        neighbors = G.neighbors(n1)
        for n2 in neighbors:
            gc += grav_centrality[n2]
        grav_centrality_plus.append(gc)
    return dict(zip(list(G.nodes()), grav_centrality_plus))

def improved_gravity_centrality2(G, path_length_dict, degree_dict, i_kshell_dict):
    # https://www.nature.com/articles/s41598-021-01218-1
    DK = [] # "Identifying influential spreaders in complex networks by an improved gravity model"
    for n1, attrs in G.nodes(data=True):
        DK.append(degree_dict[n1] + i_kshell_dict[n1])
    DK_dict = dict(zip(list(G.nodes()), DK))
    DKGM = []
    for n1, attrs in G.nodes(data=True):
        tmp = 0
        for n2, attrs in G.nodes(data=True):
            if n1 in path_length_dict and n2 in path_length_dict[n1]: 
                if path_length_dict[n1][n2] <= 3 and n1 != n2:
                    node_gravity = (DK_dict[n1] * DK_dict[n2]) / (path_length_dict[n1][n2] ** 2)
                    tmp += node_gravity
                else:
                    continue
            else:
                continue
        DKGM.append(tmp)
    return dict(zip(list(G.nodes()), DKGM))

def improved_gravity_centrality2_plus(G, improved_grav_centrality2):
    im_grav_centrality_plus2 = []
    
    for n1, attrs in G.nodes(data=True):
        gc = 0
        neighbors = G.neighbors(n1)
        for n2 in neighbors:
            gc += improved_grav_centrality2[n2]
        im_grav_centrality_plus2.append(gc)
    return dict(zip(list(G.nodes()), im_grav_centrality_plus2))

def improved_gravity_centrality(G, path_length_dict, core_num_dict, degree_dict):
    # https://www.sciencedirect.com/science/article/abs/pii/S0096300318303461
    grav_centrality = [] #  Improved centrality indicators to characterize the nodal spreading capability in complex networks
    for n1, attrs in G.nodes(data=True):
        gc = 0
        for n2, attrs in G.nodes(data=True):
            if n1 in path_length_dict and n2 in path_length_dict[n1]: 
                if path_length_dict[n1][n2] <= 3 and n1 != n2:
                    node_gravity = (core_num_dict[n1] * degree_dict[n2]) / (path_length_dict[n1][n2] ** 2)
                    gc += node_gravity
                else:
                    continue
            else:
                continue
        grav_centrality.append(gc)
    return dict(zip(list(G.nodes()), grav_centrality))

def improved_gravity_centrality_plus(G, improved_grav_centrality):
    im_grav_centrality_plus = []
    
    for n1, attrs in G.nodes(data=True):
        gc = 0
        neighbors = G.neighbors(n1)
        for n2 in neighbors:
            gc += improved_grav_centrality[n2]
        im_grav_centrality_plus.append(gc)
    return dict(zip(list(G.nodes()), im_grav_centrality_plus))

def local_gravity_centrality(G, path_length_dict, core_num_dict, degree_dict, avg_shortest_path):
    shortest_path_avg = round(avg_shortest_path / 2)
    local_grav_centrality = []
    for n1, attrs in G.nodes(data=True):
        gc = 0
        for n2, attrs in G.nodes(data=True):
            if n1 in path_length_dict and n2 in path_length_dict[n1]: 
                if path_length_dict[n1][n2] <= shortest_path_avg and n1 != n2:
                    node_gravity = (degree_dict[n1] * degree_dict[n2]) / (path_length_dict[n1][n2] ** 2)
                    gc += node_gravity
                else:
                    continue
            else:
                continue 
        local_grav_centrality.append(gc)
    return dict(zip(list(G.nodes()), local_grav_centrality))

def mcgm(G, path_length_dict, core_num_dict, degree_dict, eigenvec_centrality_dic, avg_shortest_path):
    shortest_path_avg = math.ceil(avg_shortest_path) # round(nx.average_shortest_path_length(G) / 2)
    local_grav_centrality = []
    dmax = max(degree_dict.values())
    dmedian = median(degree_dict.values())
    xmax = max(eigenvec_centrality_dic.values())
    xmedian = median(eigenvec_centrality_dic.values())
    kmax = max(core_num_dict.values())
    kmedian = median(core_num_dict.values())
    try:
        alpha = max((dmedian/dmax), (xmedian/xmax))  /  (kmedian/ kmax)
    except:
        alpha = 0

    for n1, attrs in G.nodes(data=True):
        gc = 0
        for n2, attrs in G.nodes(data=True):
            if n1 in path_length_dict and n2 in path_length_dict[n1]: 
                if path_length_dict[n1][n2] <= shortest_path_avg and n1 != n2:

                    if dmax != 0:
                        normalized_degree_n1 = degree_dict[n1]/dmax
                        normalized_degree_n2 = degree_dict[n2]/dmax
                    else:
                        normalized_degree_n1 = 0
                        normalized_degree_n2 = 0
                    
                    if xmax != 0:
                        normalized_eigenvec_n1 = eigenvec_centrality_dic[n1]/xmax
                        normalized_eigenvec_n2 = eigenvec_centrality_dic[n2]/xmax
                    else:
                        normalized_eigenvec_n1 = 0
                        normalized_eigenvec_n2 = 0
                    
                    if kmax != 0:
                        normalized_kcore_n1 = (alpha*core_num_dict[n1])/kmax
                        normalized_kcore_n2 = (alpha*core_num_dict[n2])/kmax
                    else:
                        normalized_kcore_n1 = 0
                        normalized_kcore_n2 = 0
                    
                    
                    mcgm_val = (normalized_degree_n1 + normalized_kcore_n1 + normalized_eigenvec_n1) * (normalized_degree_n2 + normalized_kcore_n2 + normalized_eigenvec_n2) / (path_length_dict[n1][n2] ** 2)
                    gc += mcgm_val
                else:
                    continue
            else:
                continue
        local_grav_centrality.append(gc)
    return dict(zip(list(G.nodes()), local_grav_centrality))

def density_centrality(G, path_length_dict, degree_dict):
    density_centrality = []
    for n1, attrs in G.nodes(data=True):
        val = 0
        for n2, attrs in G.nodes(data=True):
            if n1 in path_length_dict and n2 in path_length_dict[n1]: 
                if path_length_dict[n1][n2] <= 3 and n1 != n2:
                    val += degree_dict[n1] / (math.pi * (path_length_dict[n1][n2] ** 2))
                else:
                    continue
            else:
                continue    
        density_centrality.append(val)
    return dict(zip(list(G.nodes()), density_centrality))

def clustered_local_degree(G, path_length_dict, clustering_coefficient, degree_dict):
    # https://www.worldscientific.com/doi/10.1142/S0217979218501187
    clustered_local_degree = []
    for n1, attrs in G.nodes(data=True):
        val = 0
        for n2, attrs in G.nodes(data=True):
            if n1 in path_length_dict and n2 in path_length_dict[n1]: 
                if path_length_dict[n1][n2] <= 1 and n1 != n2:
                    val += degree_dict[n2]
                else:
                    continue
            else:
                continue
        clustered_local_degree.append((1+clustering_coefficient[n1]) * val)
    return dict(zip(list(G.nodes()), clustered_local_degree))

def new_measure1(G, path_length_dict, degree_dict, i_kshell_dict, avg_shortest_path):
    DK = [] # "Identifying influential spreaders in complex networks by an improved gravity model"
    for n1, attrs in G.nodes(data=True):
        DK.append(degree_dict[n1] + i_kshell_dict[n1])
    DK_dict = dict(zip(list(G.nodes()), DK))
    DKGM = []
    for n1, attrs in G.nodes(data=True):
        tmp = 0
        for n2, attrs in G.nodes(data=True):
            if path_length_dict[n1][n2] <= 3 and n1 != n2:
                node_gravity = (DK_dict[n1] * DK_dict[n2]) / (path_length_dict[n1][n2])
                tmp += node_gravity
            else:
                continue
        DKGM.append(tmp)
    dkninl_dict0 = dict(zip(list(G.nodes()), DKGM))

    ninl1 = []
    ninl2 = []
    ninl3 = []
    # for layer in range(3):
    for n1, attrs in G.nodes(data=True):
        tmp = 0
        for n2, attrs in G.nodes(data=True):
            if path_length_dict[n1][n2] <= 1 and n1 != n2:
                tmp += dkninl_dict0[n2]
        ninl1.append(tmp)
    dkninl_dict1 = dict(zip(list(G.nodes()), ninl1))
    for n1, attrs in G.nodes(data=True):
        tmp = 0
        for n2, attrs in G.nodes(data=True):
            if path_length_dict[n1][n2] <= 1 and n1 != n2:
                tmp += dkninl_dict1[n2]
        ninl2.append(tmp)
    dkninl_dict2 = dict(zip(list(G.nodes()), ninl2))
    for n1, attrs in G.nodes(data=True):
        tmp = 0
        for n2, attrs in G.nodes(data=True):
            if path_length_dict[n1][n2] <= 1 and n1 != n2:
                tmp += dkninl_dict2[n2]
        ninl3.append(tmp)
    dkninl_dict3 = dict(zip(list(G.nodes()), ninl3))
    return dkninl_dict3

def new_measure2(G, path_length_dict, degree_dict, i_kshell_dict, avg_shortest_path):
    DK = [] # "Identifying influential spreaders in complex networks by an improved gravity model"
    for n1, attrs in G.nodes(data=True):
        DK.append(degree_dict[n1] + i_kshell_dict[n1])
    DK_dict = dict(zip(list(G.nodes()), DK))
    DKGM = []
    for n1, attrs in G.nodes(data=True):
        tmp = 0
        for n2, attrs in G.nodes(data=True):
            if path_length_dict[n1][n2] <= 3 and n1 != n2:
                node_gravity = (DK_dict[n1] * DK_dict[n2]) / (path_length_dict[n1][n2] ** 2)
                tmp += node_gravity
            else:
                continue
        DKGM.append(tmp)
    dkninl_dict0 = dict(zip(list(G.nodes()), DKGM))

    ninl1 = []
    ninl2 = []
    ninl3 = []
    # for layer in range(3):
    for n1, attrs in G.nodes(data=True):
        tmp = 0
        for n2, attrs in G.nodes(data=True):
            if path_length_dict[n1][n2] <= 1 and n1 != n2:
                tmp += dkninl_dict0[n2]
        ninl1.append(tmp)
    dkninl_dict1 = dict(zip(list(G.nodes()), ninl1))
    for n1, attrs in G.nodes(data=True):
        tmp = 0
        for n2, attrs in G.nodes(data=True):
            if path_length_dict[n1][n2] <= 1 and n1 != n2:
                tmp += dkninl_dict1[n2]
        ninl2.append(tmp)
    dkninl_dict2 = dict(zip(list(G.nodes()), ninl2))
    for n1, attrs in G.nodes(data=True):
        tmp = 0
        for n2, attrs in G.nodes(data=True):
            if path_length_dict[n1][n2] <= 1 and n1 != n2:
                tmp += dkninl_dict2[n2]
        ninl3.append(tmp)
    dkninl_dict3 = dict(zip(list(G.nodes()), ninl3))
    return dkninl_dict3

def new_measure3(G, path_length_dict, degree_dict, i_kshell_dict, avg_shortest_path):
    DK = [] # "Identifying influential spreaders in complex networks by an improved gravity model"
    for n1, attrs in G.nodes(data=True):
        DK.append(degree_dict[n1] + i_kshell_dict[n1])
    DK_dict = dict(zip(list(G.nodes()), DK))
    DKGM = []
    for n1, attrs in G.nodes(data=True):
        tmp = 0
        for n2, attrs in G.nodes(data=True):
            if path_length_dict[n1][n2] <= 3 and n1 != n2:
                node_gravity = (DK_dict[n1] * DK_dict[n2])
                tmp += node_gravity
            else:
                continue
        DKGM.append(tmp)
    dkninl_dict0 = dict(zip(list(G.nodes()), DKGM))

    ninl1 = []
    ninl2 = []
    ninl3 = []
    # for layer in range(3):
    for n1, attrs in G.nodes(data=True):
        tmp = 0
        for n2, attrs in G.nodes(data=True):
            if path_length_dict[n1][n2] <= 1 and n1 != n2:
                tmp += dkninl_dict0[n2]
        ninl1.append(tmp)
    dkninl_dict1 = dict(zip(list(G.nodes()), ninl1))
    for n1, attrs in G.nodes(data=True):
        tmp = 0
        for n2, attrs in G.nodes(data=True):
            if path_length_dict[n1][n2] <= 1 and n1 != n2:
                tmp += dkninl_dict1[n2]
        ninl2.append(tmp)
    dkninl_dict2 = dict(zip(list(G.nodes()), ninl2))
    for n1, attrs in G.nodes(data=True):
        tmp = 0
        for n2, attrs in G.nodes(data=True):
            if path_length_dict[n1][n2] <= 1 and n1 != n2:
                tmp += dkninl_dict2[n2]
        ninl3.append(tmp)
    dkninl_dict3 = dict(zip(list(G.nodes()), ninl3))
    return dkninl_dict3

def new_measure4(G, path_length_dict, degree_dict, i_kshell_dict, avg_shortest_path):
    DK = [] # "Identifying influential spreaders in complex networks by an improved gravity model"
    for n1, attrs in G.nodes(data=True):
        tmp = 0
        for n2, attrs in G.nodes(data=True):
            if path_length_dict[n1][n2] <= 3 and n1 != n2:
                tmp += i_kshell_dict[n1] * degree_dict[n2] 
        DK.append(tmp)
    DK_dict = dict(zip(list(G.nodes()), DK))
    DKGM = []
    for n1, attrs in G.nodes(data=True):
        tmp = 0
        for n2, attrs in G.nodes(data=True):
            if path_length_dict[n1][n2] <= 3 and n1 != n2:
                node_gravity = (DK_dict[n1] * DK_dict[n2]) / (path_length_dict[n1][n2])
                tmp += node_gravity
            else:
                continue
        DKGM.append(tmp)
    dkninl_dict0 = dict(zip(list(G.nodes()), DKGM))

    ninl1 = []
    ninl2 = []
    ninl3 = []
    # for layer in range(3):
    for n1, attrs in G.nodes(data=True):
        tmp = 0
        for n2, attrs in G.nodes(data=True):
            if path_length_dict[n1][n2] <= 1 and n1 != n2:
                tmp += dkninl_dict0[n2]
        ninl1.append(tmp)
    dkninl_dict1 = dict(zip(list(G.nodes()), ninl1))
    for n1, attrs in G.nodes(data=True):
        tmp = 0
        for n2, attrs in G.nodes(data=True):
            if path_length_dict[n1][n2] <= 1 and n1 != n2:
                tmp += dkninl_dict1[n2]
        ninl2.append(tmp)
    dkninl_dict2 = dict(zip(list(G.nodes()), ninl2))
    for n1, attrs in G.nodes(data=True):
        tmp = 0
        for n2, attrs in G.nodes(data=True):
            if path_length_dict[n1][n2] <= 1 and n1 != n2:
                tmp += dkninl_dict2[n2]
        ninl3.append(tmp)
    dkninl_dict3 = dict(zip(list(G.nodes()), ninl3))
    return dkninl_dict3

def new_measure5(G, path_length_dict, degree_dict, i_kshell_dict, avg_shortest_path):
    DK = [] # "Identifying influential spreaders in complex networks by an improved gravity model"
    for n1, attrs in G.nodes(data=True):
        tmp = 0
        for n2, attrs in G.nodes(data=True):
            if path_length_dict[n1][n2] <= 3 and n1 != n2:
                tmp += (i_kshell_dict[n1] * degree_dict[n2]) / (path_length_dict[n1][n2])
        DK.append(tmp)
    DK_dict = dict(zip(list(G.nodes()), DK))
    DKGM = DK

    dkninl_dict0 = dict(zip(list(G.nodes()), DKGM))

    return dkninl_dict0

def ninl(G, degree_dict, path_length_dict, avg_shortest_path):
    L = math.ceil(avg_shortest_path)
    ninl = []
    for n1, attrs in G.nodes(data=True):
        tmp = 0
        for n2, attrs in G.nodes(data=True):
            if n1 in path_length_dict and n2 in path_length_dict[n1]: 
                try:
                    if path_length_dict[n1][n2] <= L and n1 != n2:
                        tmp += degree_dict[n2]
                    else:
                        continue
                except:
                    print(n1)
                    print(n2)
                    print(path_length_dict)
            else:
                continue
        val = degree_dict[n1] + tmp
        ninl.append(val)
    ninl_dict0 = dict(zip(list(G.nodes()), ninl))

    ninl1 = []
    ninl2 = []
    ninl3 = []
    # for layer in range(3):
    for n1, attrs in G.nodes(data=True):
        tmp = 0
        for n2, attrs in G.nodes(data=True):
            if path_length_dict[n1][n2] <= 1 and n1 != n2:
                tmp += ninl_dict0[n2]
        ninl1.append(tmp)
    ninl_dict1 = dict(zip(list(G.nodes()), ninl1))
    for n1, attrs in G.nodes(data=True):
        tmp = 0
        for n2, attrs in G.nodes(data=True):
            if path_length_dict[n1][n2] <= 1 and n1 != n2:
                tmp += ninl_dict1[n2]
        ninl2.append(tmp)
    ninl_dict2 = dict(zip(list(G.nodes()), ninl2))
    for n1, attrs in G.nodes(data=True):
        tmp = 0
        for n2, attrs in G.nodes(data=True):
            if path_length_dict[n1][n2] <= 1 and n1 != n2:
                tmp += ninl_dict2[n2]
        ninl3.append(tmp)
    ninl_dict3 = dict(zip(list(G.nodes()), ninl3))
    
    return ninl_dict3

def i_kshell(G, core_num_dict, core_num_iteration_dict):
    iks = []
    for n, attrs in G.nodes(data=True):
        iks.append(core_num_dict[n] + core_num_iteration_dict[n])
    return dict(zip(list(G.nodes()), iks))

def k_shell_alternative(G):
        # Copy the graph
    h = G.copy()
    it = 1
        
    # Bucket being filled currently
    tmp = []
        
    # list of lists of buckets
    buckets = []
    kshell_iteration_dict = {}
    count_kshell_iteration = 0

    while (1):
        count_kshell_iteration += 1
        flag = check(h, it)
        if (flag == 0):
            it += 1
            buckets.append(tmp)
            tmp = []
        if (flag == 1):
            node_set = find_nodes(h, it)
            for each in node_set:
                h.remove_node(each)
                tmp.append(each)
                kshell_iteration_dict[each] = count_kshell_iteration
        if (h.number_of_nodes() == 0):
            buckets.append(tmp)
            break

    k_shell_dict = {}
    for n1, attrs in G.nodes(data=True):
        for count, shell in enumerate(buckets):
            if n1 in shell:
                k_shell_dict[n1] = count + 1
                break
    return k_shell_dict, kshell_iteration_dict

def gli_method(G, i_kshell_dict, degree_dict, path_length_dict):
    # https://iopscience.iop.org/article/10.1088/1674-1056/ab969f/pdf
    gli_dict = {}
    for n1, attrs in G.nodes(data=True):
        tmp1 = (i_kshell_dict[n1] + degree_dict[n1]) / (sum([i_kshell_dict[cur_n] + degree_dict[cur_n] for cur_n, _ in list(G.nodes(data=True))]))
        tmp2 = []
        for n2, attrs in G.nodes(data=True):
            if n1 in path_length_dict and n2 in path_length_dict[n1]: 
                if path_length_dict[n1][n2] <= 3 and n1 != n2:
                    tmp2.append((i_kshell_dict[n2] + degree_dict[n2]) / path_length_dict[n1][n2])
                else:
                    continue
            else:
                continue
        gli_dict[n1] = math.exp(tmp1) * sum(tmp2)
    return gli_dict            

def ks_if(G, core_num_dict, core_num_iteration_dict, degree_dict):
    max_iter = max(core_num_iteration_dict.values())
    s_ni =  {}
    ks_if = {}
    for n, attrs in G.nodes(data=True):
        s_ni[n] = core_num_dict[n] * (1 + (core_num_iteration_dict[n]) / max_iter)
    
    nsd_dict = {}
    
    for n1, attrs in G.nodes(data=True):
        friends = [n2 for n2 in G.neighbors(n1)]
        tmp_ns = 0
        for friend in friends:
            tmp_ns += s_ni[friend] * degree_dict[friend]
        nsd_dict[n1] = tmp_ns

    for n, attrs in G.nodes(data=True):
        ks_if[n] = (s_ni[n] * degree_dict[n]) + nsd_dict[n]
    
    return ks_if

def mcde(G, core_num_dict, degree_dict,):
    MCDE_dict = {}
    MCDWE_dict = {}
    max_core = max(core_num_dict.values())
    sorted_cores = sorted(set(core_num_dict.values()))
    alpha = 1
    beta = 1 
    gamma = 1
    
    entropy_dict = {}
    w_entropy_dict = {}
    for n1, attrs in G.nodes(data=True):
        friends = [n2 for n2 in G.neighbors(n1)]
        p_dict = {}
        for c in sorted_cores:
            num_friends_in_core = 0
            for friend in friends:
                if core_num_dict[friend] == c:
                    num_friends_in_core += 1
            p_dict[c] = num_friends_in_core / len(friends)

        cur_entropy = 0 
        cur_w_entropy = 0 
        
        for c in sorted_cores:
            try:
                cur_entropy += p_dict[c] * math.log2(p_dict[c])
                cur_w_entropy += (1 / (max_core - c + 1)) * (p_dict[c] * math.log2(p_dict[c]))
            except:
                cur_entropy = 0
                cur_w_entropy = 0
        entropy_dict[n1] = -cur_entropy
        w_entropy_dict[n1] = -cur_w_entropy

    for n1, attrs in G.nodes(data=True):
        MCDE = alpha * core_num_dict[n1] + beta * degree_dict[n1] + gamma * entropy_dict[n1]
        MCDWE = alpha * core_num_dict[n1] + beta * degree_dict[n1] + gamma * w_entropy_dict[n1]
        MCDE_dict[n1] = MCDE
        MCDWE_dict[n1] = MCDWE
    
    return MCDE_dict, MCDWE_dict

def erm(G, degree_dict,):
    # https://www.sciencedirect.com/science/article/abs/pii/S0960077917303788
    d1 = {}
    for n1, attrs in G.nodes(data=True):
        friends = [n2 for n2 in G.neighbors(n1)]
        d1[n1] = sum([degree_dict[friend] for friend in friends])

    d2 = {}
    for n1, attrs in G.nodes(data=True):
        friends = [n2 for n2 in G.neighbors(n1)]
        d2[n1] = sum([d1[friend] for friend in friends])
    
    E1 = {}
    E2 = {}
    lamda = {}
    for n1, attrs in G.nodes(data=True):
        friends = [n2 for n2 in G.neighbors(n1)]
        tmp1 = -sum([(degree_dict[n2] / d1[n1]) * math.log(degree_dict[n2] / d1[n1]) for n2 in friends])
        tmp2 = -sum([(d1[n2] / d2[n1]) * math.log(d1[n2] / d2[n1]) for n2 in friends])
        E1[n1] = tmp1
        E2[n1] = tmp2 
    
    EC = {}
    for n1, attrs in G.nodes(data=True):
        friends = [n2 for n2 in G.neighbors(n1)]
        if max(E2.values()) > 0:
            lamda[n1] = E2[n1] / (max(E2.values()))
        else:
            lamda[n1] = 0
        EC[n1] = sum([E1[friend] + lamda[n1] * E2[friend] for friend in friends])
    
    SI = {}
    for n1, attrs in G.nodes(data=True):
        friends = [n2 for n2 in G.neighbors(n1)]
        SI[n1] = sum([EC[friend] for friend in friends])
    
    ERM = {}
    for n1, attrs in G.nodes(data=True):
        friends = [n2 for n2 in G.neighbors(n1)]
        ERM[n1] = sum([SI[friend] for friend in friends])

    return ERM

def new_gli_method(G, core_num_dict, degree_dict,):
    # https://www.nature.com/articles/s41598-022-26984-4#Sec4
    jacc_dict = {}
    for n1, attrs in G.nodes(data=True):
        n1_friends = [node for node in G.neighbors(n1)]
        for n2, attrs in G.nodes(data=True):
            if n1 == n2:
                continue
            n2_friends = [node for node in G.neighbors(n2)]
            jacc_dict[f"{n1}-{n2}"] = jaccard(n1_friends, n2_friends)

    omega = {}
    for n1, attrs in G.nodes(data=True):
        friends = [node for node in G.neighbors(n1)]
        tmp = 0
        for friend in friends:
            tmp += degree_dict[friend] * jacc_dict[f"{n1}-{friend}"] + core_num_dict[friend]
        omega[n1] = tmp

    new_gli = {}
    for n1, attrs in G.nodes(data=True):
        local = degree_dict[n1] + (omega[n1] / max(degree_dict.values()))
        new_gli[n1] = local + core_num_dict[n1]
    
    return new_gli

def h_index(G):
    h_index_dict = {}
    for n1, attrs in G.nodes(data=True):
        sorted_neighbor_degrees = sorted((G.degree(v) for v in G.neighbors(n1)), reverse=True)
        h = 0
        for i in range(1, len(sorted_neighbor_degrees)+1):
            if sorted_neighbor_degrees[i-1] < i:
                break
            h = i
        h_index_dict[n1] = h

    return h_index_dict

def local_h_index(G, h_index_dict):
    # https://www.sciencedirect.com/science/article/abs/pii/S0378437118309932
    local_h_index_dict = {}
    for n1, attrs in G.nodes(data=True):
        friends = [n2 for n2 in G.neighbors(n1)]
        local_h_index_dict[n1] = h_index_dict[n1] + sum([h_index_dict[n2] for n2 in friends])
    
    return local_h_index_dict

def dsr(G, degree_dict, core_num_dict, core_num_iteration_dict):
    # https://www.sciencedirect.com/science/article/abs/pii/S0167739X18319009
    H1_dict = {}
    IKS_prime = {}
    for n1, attrs in G.nodes(data=True):
        friends = [n2 for n2 in G.neighbors(n1)]
        res = 0
        for friend in friends:
            tmp1 = core_num_iteration_dict[friend] / sum([core_num_iteration_dict[f] for f in friends])
            tmp2 = math.log(core_num_iteration_dict[friend] / sum([core_num_iteration_dict[f] for f in friends]))
            res = -(tmp1 * tmp2)
        
        IKS_prime[n1] = sum([core_num_iteration_dict[f] for f in friends]) / len(friends)
        H1_dict[n1] = res

    sorted_cores = sorted(set(core_num_dict.values()))
    max_core = max(core_num_dict.values())
    X_vecs = {}
    for n1, attrs in G.nodes(data=True):
        friends = [n2 for n2 in G.neighbors(n1)]
        p_vals = []
        for c in sorted_cores:
            num_friends_in_core = 0
            for friend in friends:
                if core_num_dict[friend] == c:
                    num_friends_in_core += 1
            p_vals.append(num_friends_in_core / len(friends))
        X_vecs[n1] = p_vals

    JSD_dict = {}
    for n1, attrs in G.nodes(data=True):
        friends = [n2 for n2 in G.neighbors(n1)]       
        part1 = []
        # part2 = []
        for core in range(len(sorted_cores)):
            tmp1 = 0
            tmp2 = 0
            for friend in friends:
                # tmp1 += (1/degree_dict[n1]) * X_vecs[n1]
                tmp1 += (1/degree_dict[n1]) * X_vecs[friend][core]
                # if core == 0:
                tmp2 += (1/degree_dict[n1]) * entropy(X_vecs[friend])
            part1.append(tmp1)
            # part2.append(tmp2)
        part1 = entropy(part1)
        part2 = tmp2

        JSD_dict[n1] = part1 - part2
    
    dsc_dict = {}
    for n1, attrs in G.nodes(data=True):
        dsc_dict[n1] = IKS_prime[n1] * H1_dict[n1] * JSD_dict[n1]
    
    dsr_dict = {}
    for n1, attrs in G.nodes(data=True):
        friends = [n2 for n2 in G.neighbors(n1)]
        dsr_dict[n1] = sum([dsc_dict[friend] for friend in friends])

    edsr_dict = {}
    for n1, attrs in G.nodes(data=True):
        friends = [n2 for n2 in G.neighbors(n1)]
        edsr_dict[n1] = sum([dsr_dict[friend] for friend in friends])
    
    return dsr_dict, edsr_dict

def LS(G, degree_dict, core_num_dict):
    # https://www.sciencedirect.com/science/article/abs/pii/S0378437118310707
    ls_dict = {}
    for n1, n2 in list(G.edges()):
        friends1 = set([n for n in G.neighbors(n1)])
        friends2 = set([n for n in G.neighbors(n2)])
        intersec = friends1.intersection(friends2) 
        union = friends1.union(friends2) 
        ls_dict[f"{n1}-{n2}"] = 1 - (len(intersec) / len(union))
    
    ks_prime_dict = {}
    for n1 in list(G.nodes()):
        friends = set([n for n in G.neighbors(n1)])
        tmp1 = 0
        for friend in friends:
            if f"{friend}-{n1}" in ls_dict:
                tmp1 += ls_dict[f"{friend}-{n1}"]
            else:
                tmp1 += ls_dict[f"{n1}-{friend}"]
        ks_prime_dict[n1] = tmp1 / degree_dict[n1] * core_num_dict[n1]
    
    I_dict = {}
    for n1 in list(G.nodes()):
        friends = set([n for n in G.neighbors(n1)])
        tmp1 = 0
        for friend in friends:
            if f"{friend}-{n1}" in ls_dict:
                tmp1 += ls_dict[f"{friend}-{n1}"] * ks_prime_dict[friend]
            else:
                tmp1 += ls_dict[f"{n1}-{friend}"] * ks_prime_dict[friend]
        I_dict[n1] = tmp1 / degree_dict[n1] * core_num_dict[n1]
    return I_dict

def ECRM(G, degree_dict, core_num_iteration_dict):
    sv_dict = {}
    sorted_core_iters = sorted(set(core_num_iteration_dict.values()))
    max_core_iter = max(core_num_iteration_dict.values())
    for n1, attrs in G.nodes(data=True):
        friends = [n2 for n2 in G.neighbors(n1)]
        p_vals = []
        for c in sorted_core_iters:
            num_friends_in_core = 0
            for friend in friends:
                if core_num_iteration_dict[friend] == c:
                    num_friends_in_core += 1
            p_vals.append(num_friends_in_core)
        sv_dict[n1] = p_vals
    
    C_dict = {}
    for n1, attrs in G.nodes(data=True):
        c_friends = {}
        friends = [n2 for n2 in G.neighbors(n1)]  
        for friend in friends:
            tmp1 = 0
            tmp2 = 0
            for c in range(len(sorted_core_iters)): 
                tmp1 += (sv_dict[n1][c] - degree_dict[n1] / max_core_iter) * (sv_dict[friend][c] - degree_dict[friend] / max_core_iter)
                tmp2 += math.sqrt(((sv_dict[n1][c] - degree_dict[n1] / max_core_iter) ** 2)) * math.sqrt(((sv_dict[friend][c] - degree_dict[friend] / max_core_iter) ** 2))
            try:
                c_friends[friend] = (tmp1 / tmp2)
            except:
                c_friends[friend] = 0
        C_dict[n1] = c_friends

    SCC_dict = {}
    for n1, attrs in G.nodes(data=True):
        friends = [n2 for n2 in G.neighbors(n1)]
        tmp1 = 0
        for friend in friends:
            tmp1 += (2 - C_dict[n1][friend]) + ((2 * (degree_dict[friend]) / max(degree_dict.values())) + 1)
        SCC_dict[n1] = tmp1
    
    CRM_dict = {}
    for n1, attrs in G.nodes(data=True):
        friends = [n2 for n2 in G.neighbors(n1)]
        tmp1 = 0
        for friend in friends:
            tmp1 += SCC_dict[friend]
        CRM_dict[n1] = tmp1
        
    ECRM_dict = {}
    for n1, attrs in G.nodes(data=True):
        friends = [n2 for n2 in G.neighbors(n1)]
        tmp1 = 0
        for friend in friends:
            tmp1 += CRM_dict[friend]
        ECRM_dict[n1] = tmp1
    
    return ECRM_dict
        


## Utils
def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def entropy(X):
    res = 0
    for x in X:
        try:
            res += x * math.log(x)
        except:
            continue
    return -res

def get_tau_c(graph_list, graph_name_list, additional_taus=True):
    average_degrees = []
    second_order_moments = []
    for graph in graph_list:
        degrees = [val for (node, val) in graph.degree()]
        av = mean(degrees)
        average_degrees.append(av)
        second_order_moments.append(nth_moment(graph, 2))
        # second_order_moments.append(np.var(degrees))

    list_tau_c = [a/(m-a) for a, m in zip(average_degrees, second_order_moments)]
    add_taus = []
    tau_set = []
    if additional_taus:
        for tau_c in list_tau_c:
            tau_set.append([tau_c * 0.5, tau_c * 0.75, tau_c, tau_c * 1.25, tau_c *1.5, tau_c *1.75, tau_c *2.0, tau_c *2.25, tau_c *2.5 ]) # 
        return list_tau_c, tau_set
    else:
        return list_tau_c

def nth_moment(g,n):
    s = 0
    for node in g.nodes:
        s += g.degree[node] ** n
    return (s/len(g))
    
# Find list of nodes with particular degree
def find_nodes(h, it):
    set1 = []
    for i in h.nodes():
        if (h.degree(i) <= it):
            set1.append(i)
    return set1
  
def check(h, d):
    f = 0  # there is no node of deg <= d
    for i in h.nodes():
        if (h.degree(i) <= d):
            f = 1
            break
    return f

# Simulation
def run_sim_single_graph(G, graph_name, 
                        num_classes, num_simulations_per_network, 
                        num_simulations_per_node,
                        num_infected_nodes_list, tau_list, 
                        gamma_list, tmax_list,
                        save_graph_as_file=False,
                        K_range=3):
    # Setup graph
    graph_set_name = f"graph_set_{graph_name}"
    all_nodes = list(G.nodes())
    node_dict = dict(zip(all_nodes, [0] * len(all_nodes)))
    num_nodes = len(all_nodes)
    pos = {node:node for node in G}
    sim_kwargs = {'pos': pos}
    _G = G.copy()
    parameter_combinations = list(itertools.product(num_infected_nodes_list, tau_list, gamma_list, tmax_list))
    
    save_tau = 0
    simulation_runs = []
    influence_node_dict = dict(zip(all_nodes, [0] * len(all_nodes)))
    number_of_sims = 0
    for num_infected_nodes, tau, gamma, tmax in tqdm(parameter_combinations, total=len(parameter_combinations), desc="Simulation run of current infected network"):
        previous_initial_infected_nodes = []
        save_tau = tau
        # for sim_run in range(num_simulations_per_network):
        #     initial_infected_nodes = random.sample(all_nodes, num_infected_nodes) # [x for x in all_nodes if x not in previous_initial_infected_nodes]
        #     previous_initial_infected_nodes.extend(initial_infected_nodes)
        #     sim = EoN.fast_SIR(_G, tau, gamma, initial_infecteds=initial_infected_nodes,
        #                 return_full_data=True, tmax=tmax,
        #                 # transmission_weight="encounters",
        #                 sim_kwargs=sim_kwargs)
        #     simulation_runs.append(sim)
        #     if len(previous_initial_infected_nodes) >= num_nodes:
        #         previous_initial_infected_nodes = []
        for n in tqdm(list(_G.nodes()) * num_simulations_per_node, total=len(list(_G.nodes()) * num_simulations_per_node), desc=f"Using each node as initial node in sim"):
            for sim_run in range(num_simulations_per_network):
                sim = EoN.fast_SIR(_G, tau, gamma, initial_infecteds=n,
                            return_full_data=True,# tmax=tmax,
                            # transmission_weight="encounters",
                            sim_kwargs=sim_kwargs)
                node_dict = get_num_infections_per_node(nodes=all_nodes, sim_run=sim, node_dict=node_dict, init_node=n)
                influence_node_dict[n] += get_num_infected_nodes(nodes=all_nodes, sim_run=sim, init_node=n) / len(list(_G.nodes()))
                # simulation_runs.append(sim)

                number_of_sims += 1




    # Get number of infections per node, create labeled graph
    for key, value in influence_node_dict.items():
        influence_node_dict[key] = value / num_simulations_per_network
    G_labeled = create_node_labels(_G, node_dict, influence_node_dict, number_of_sims)
    
    if save_graph_as_file:
        # Get node features
        create_node_attributes(G_labeled, K_range=K_range)
        # print(list(set(list(attrs))))
        # Store labeled graphs in dir
        path = f"{graph_dir}{graph_set_name}/raw_dir/"
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory '{path}' was created!")
        nx.write_gpickle(G_labeled, path + f"graph_{graph_name}_tau{save_tau}.pkl", protocol=4)

    # parameter = {
    #     f"graph_{graph_name}" : {
    #         # "graph_name" : ,
    #         "tmax" : tmax,
    #         "tau" : tau,
    #         "gamma" : gamma,
    #         "num_simulations_per_network" : num_simulations_per_network,
    #         "num_nodes" : num_nodes,
    #         "num_infected_nodes" : num_infected_nodes
    #                     }
    #             }
    return G_labeled, simulation_runs

def run_sim_mult_graphs(G_raw, graph_name,
                        taus = [0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375],
                        num_simulations_per_network=10,
                        num_simulations_per_node=1,
                        save_graphs_as_file=False,
                        K_range=3):
    print(f"Run simulation for {graph_name}")
    print(f"Number of nodes: {len(list(G_raw.nodes()))}")
    graph_list = []
    for tau in taus:
        list_tau = [tau]
        list_tmax = [50] # 20
        list_gamma = [1.0]
        list_num_simulations_per_network = num_simulations_per_network
        list_num_simulations_per_node = num_simulations_per_node
        list_num_infected_nodes = [1]
        list_num_classes = 10
        parameter_list = [
                        graph_name, list_num_classes, list_num_simulations_per_network,
                        list_num_simulations_per_node,
                        list_num_infected_nodes, list_tau,
                        list_gamma, list_tmax,                
                        ]

        # G_raw = create_french_school_graph()
        G_cur = G_raw.copy()
        G_labeled, simulation_runs = run_sim_single_graph(G_cur, *parameter_list, save_graph_as_file=save_graphs_as_file, K_range=K_range)
        graph_list.append(G_labeled)
    if len(graph_list) > 1:
        plot_class_distribution_compare_graphs(graph_list, graph_name, taus)
    return graph_list



# Plotting
def plot_class_distribution_single_graph(G, graph_set_name):
    # G = nx.read_gpickle(path + f"graph_{graph_name}.pkl")
    _num_classes = list(nx.get_node_attributes(G, "risk_raw").values())
    classes = [x for x in _num_classes]
    classes_as_ints = [int(x) for x in _num_classes]
    sorted_list = sorted(classes)
    print(list(set(sorted_list)))
    sorted_counted = Counter(sorted_list)

    range_length = list(range(int(max(classes)))) # Get the largest value to get the range.
    data_series = {}

    for i in range_length:
        data_series[i] = 0 # Initialize series so that we have a template and we just have to fill in the values.
    for key, value in sorted_counted.items():
        data_series[key] = value

    data_series = pd.Series(data_series)
    x_values = data_series.index

    fig, ax = plt.subplots()
    plt.scatter(x_values, data_series.values) # , width=0.3
    plt.xlabel("risk level")
    fig.tight_layout()
    print(f"Save figure to " + f"{plot_dir}/class_distribution/class_distribution_{graph_set_name}.png")
    fig.savefig(f"{plot_dir}/class_distribution/class_distribution_{graph_set_name}.png")#

def plot_class_distribution_compare_graphs(graph_list, graph_set_name, taus):
    fig, axs = plt.subplots(len(graph_list), 1, figsize=(15, 30))
    count = 0
    num_nodes = 0
    for g in graph_list:
        # G = nx.read_gpickle(path + f"graph_{graph_name}.pkl")
        _num_classes = list(nx.get_node_attributes(g, "risk_raw").values())
        classes = [x for x in _num_classes]
        classes_as_ints = [int(x) for x in _num_classes]
        sorted_list = sorted(classes)
        print(list(set(sorted_list)))
        sorted_counted = Counter(sorted_list)

        range_length = list(range(int(max(classes)))) # Get the largest value to get the range.
        data_series = {}

        for i in range_length:
            data_series[i] = 0 # Initialize series so that we have a template and we just have to fill in the values.
        for key, value in sorted_counted.items():
            data_series[key] = value

        data_series = pd.Series(data_series)
        x_values = data_series.index

        axs[count].scatter(x_values, data_series.values) # , width=0.3
        axs[count].set_ylabel("#nodes")
        axs[count].set_xlabel("risk of infection")
        axs[count].set_title(f"tau=={taus[count]}")
        num_nodes = len(list(g.nodes()))
        count += 1
    plt.tight_layout()
    fig.suptitle(f"{graph_set_name}, #nodes=={num_nodes}", fontsize=16)
    print(f"Save figure to " + f"{plot_dir}/class_distribution/class_distribution_comparison_{graph_set_name}.png")
    fig.savefig(f"{plot_dir}/class_distribution/class_distribution_comparison_{graph_set_name}.png")#

def draw_basic_network(g):
    # Need to create a layout when doing
    # separate calls to draw nodes and edges
    fig, ax = plt.subplots(figsize=(25, 20), dpi=120)
    risks = nx.get_node_attributes(g, "risk_raw")
    d = nx.get_node_attributes(g, "risk_raw")

    deg_centrality = nx.degree_centrality(g)
    centrality = np.fromiter(deg_centrality.values(), float)

    edge_attrs = nx.get_edge_attributes(g, "encounters")
    pos = nx.spring_layout(g, k=0.2)
    node_risks = [x*50 for x in list(risks.values())]
    nx.draw_networkx_nodes(g, pos, node_size=[x*250 for x in centrality], node_color=node_risks)
    nx.draw_networkx_edges(g, pos, width=[x / 10 for x in list(edge_attrs.values())])
    plt.legend()
    
    plt.savefig(plot_dir + "test_french")
