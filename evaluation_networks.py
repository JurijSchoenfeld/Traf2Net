import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tacoma as tc
import re
from util import plot_contact_durations, moving_average, downscale_time
from tacoma.analysis import plot_degree_distribution
import random
import contact_networks as cn
from scipy.sparse import csr_array
from scipy.stats import ks_2samp, pearsonr
import pyreadr
import teneto
import dynetx as dn
# import cdlib as cd
import networkx as nx
from collections import Counter
from math import ceil, sqrt
import util
from cdlib import algorithms, viz, evaluation, TemporalClustering
import csv
import optuna
import pickle
import shutil


# helper functions for loading saving networks, normalization and splitting them up into daily chunks
def normalized_nodes_time(df, tscale=20):
    nodes = np.union1d(df.i.unique(), df.j.unique())
    nodes_map = dict(zip(nodes, np.arange(len(nodes))))
    df.i = df.i.map(nodes_map)
    df.j = df.j.map(nodes_map)
    df.t = ((df.t - df.t.min())/20).astype('int')
    
    return df


def collect_SFHH():
    path = './data_eval/SFHH-conf-sensor.edges'
    df = pd.read_csv(path, names=['i', 'j', 't'])
    df['date'] = pd.to_datetime(df.t, unit='s')
    df['day'] = df.date.dt.date
    df = normalized_nodes_time(df)


    for grp, x in df.groupby('day'):
        x.t = x.t - x.t.min()
        x.to_parquet(f'./data_eval_split/SFHH/{grp}.parquet')


def collect_InVS():
    files = ['tij_InVS.dat', 'tij_InVS15.dat']
    path = './data_eval/'

    for i, file in enumerate(files):
        df = pd.read_csv(path + file, sep=' ', names=['t', 'i', 'j', 'Ci', 'Cj'])
        df['date'] = pd.to_datetime(df.t, unit='s')
        df['day'] = df.date.dt.date
        df = normalized_nodes_time(df)
        
        for grp, x in df.groupby('day'):
            x.t = x.t - x.t.min()
            x.to_parquet(f'./data_eval_split/InVS/f{i}_{grp}.parquet')


def collect_primaryschool():
    path = './data_eval/primaryschool.csv'
    df = pd.read_csv(path, sep='\t', names=['t', 'i', 'j', 'Ci', 'Cj'])
    df['date'] = pd.to_datetime(df.t, unit='s')
    df['day'] = df.date.dt.date
    df = normalized_nodes_time(df)

    for grp, x in df.groupby('day'):
            x.t = x.t - x.t.min()
            x.to_parquet(f'./data_eval_split/primaryschool/{grp}.parquet')


def collect_supermarked():
    path = './data_eval/data-shared/'

    for i, file in enumerate(os.listdir(path)):
        df = pyreadr.read_r(path + file)[None]

        # Get unique nodes
        unique_nodes = np.unique(df[['reporting_id', 'opposing_id']].values.flatten())
        inds = np.arange(0, len(unique_nodes), 1)

        # Extract date, time1, and time2 from each string in the array
        date_time_data = [entry.split('_')[1:4] for entry in unique_nodes]

        # Convert date, time1, and time2 to a pandas DataFrame
        df_trajectories = pd.DataFrame(date_time_data, columns=['date', 'time1', 'time2'])

        # Combine date and time columns, then convert to datetime format
        df_trajectories['activity_start_min'] = pd.to_datetime(df_trajectories['date'] + ' ' + df_trajectories['time1'], format='%Y-%m-%d %H:%M:%S')
        df_trajectories['activity_end_min'] = pd.to_datetime(df_trajectories['date'] + ' ' + df_trajectories['time2'], format='%Y-%m-%d %H:%M:%S')

        # Drop unnecessary columns
        df_trajectories.drop(['date', 'time1', 'time2'], axis=1, inplace=True)
        df_trajectories['p_id'] = np.arange(len(unique_nodes))


        # Get index like node names
        node_int_dict = dict(zip(unique_nodes, inds))

        # Map nodes to index
        df['i'] = df.reporting_id.map(node_int_dict)
        df['j'] = df.opposing_id.map(node_int_dict)

        # Extract start and end times from reporting_id and opposing_id
        df[['i_date', 'i_start', 'i_end']] = df['reporting_id'].str.split('_', expand=True).iloc[:, [1, 2, 3]]
        df[['j_date', 'j_start', 'j_end']] = df['opposing_id'].str.split('_', expand=True).iloc[:, [1, 2, 3]]

        # Combine date with start and end times
        df['i_start'] = pd.to_datetime(df['i_date'] + ' ' + df['i_start'])
        df['i_end'] = pd.to_datetime(df['i_date'] + ' ' + df['i_end'])
        df['j_start'] = pd.to_datetime(df['j_date'] + ' ' + df['j_start'])
        df['j_end'] = pd.to_datetime(df['j_date'] + ' ' + df['j_end'])

        # Drop intermediate date columns
        df.drop(['i_date', 'j_date'], axis=1, inplace=True)

        # Find the minimum timestamp
        min_timestamp = df[['i_start', 'i_end', 'j_start', 'j_end', 'timestamp']].min().min()

        # Normalize temporal columns to seconds since the smallest occurred time
        df['i_start_seconds'] = (df['i_start'] - min_timestamp).dt.total_seconds().astype('int')
        df['i_end_seconds'] = (df['i_end'] - min_timestamp).dt.total_seconds().astype('int')
        df['j_start_seconds'] = (df['j_start'] - min_timestamp).dt.total_seconds().astype('int')
        df['j_end_seconds'] = (df['j_end'] - min_timestamp).dt.total_seconds().astype('int')
        df['t'] = (df['timestamp'] - min_timestamp).dt.total_seconds().astype('int')
        df_trajectories['activity_start_min'] = (df_trajectories['activity_start_min'] - min_timestamp).dt.total_seconds().astype('int')
        df_trajectories['activity_end_min'] = (df_trajectories['activity_end_min'] - min_timestamp).dt.total_seconds().astype('int')

        df['date'] = pd.to_datetime(df.timestamp, unit='s')
        df['day'] = df.date.dt.date 

        # Remove double edges
        # Sort 'i' and 'j' in each row to make them interchangeable
        df[['i', 'j']] = np.sort(df[['i', 'j']], axis=1)

        # Create a mask for duplicated rows considering 'i' and 'j' as interchangeable
        mask = df.duplicated(subset=['i', 'j', 't'])

        # Apply the mask to keep only non-duplicated rows
        df = df[mask]

        # Keep only contacts with distance <= 2m
        df = df[df.distance <= 200.0].reset_index(drop=True)

        df.to_parquet(f'./data_eval_split/supermarked/f{i}_{df.loc[0].day}.parquet')   
        df_trajectories.to_parquet(f'./data_eval_split/supermarked/f{i}_{df.loc[0].day}_trajectories.parquet')


def collect_gallery():
    path = './data_eval/infectious/'

    for i, file in enumerate(os.listdir(path)):
        df = pd.read_csv(path + file, sep='\t', names=['t', 'i', 'j'])
        df['date'] = pd.to_datetime(df.t, unit='s')
        df['day'] = df.date.dt.date 
        df = normalized_nodes_time(df)   

        df.to_parquet(f'./data_eval_split/gallery/f{i}_{df.loc[0].day}.parquet')   


def collect_highschool():
    files = ['High-School_data_2013.csv', 'thiers_2012.csv', 'thiers_2011.csv']
    path = './data_eval/'

    for i, file in enumerate(files):
        if i == 0:
            sep = ' '
        else:
            sep = '\t'

        df = pd.read_csv(path + file, sep=sep, names=['t', 'i', 'j', 'Ci', 'Cj'])
        df['date'] = pd.to_datetime(df.t, unit='s')
        df['day'] = df.date.dt.date
        df = normalized_nodes_time(df)

        for grp, x in df.groupby('day'):
            x.t = x.t - x.t.min()
            x.to_parquet(f'./data_eval_split/highschool/f{i}_{grp}.parquet')


class EvaluationNetwork:
    def __init__(self, name, TU, temporal_offset, path=None, new_time=None, drop_groups=None, switch_off_time=None):
        self.name = name
        self.TU = TU
        self.temporal_offset = temporal_offset
        self.TC_EMP = None

        # Available methods for evaluation
        self.method_func = {'avg_internal_degree': evaluation.average_internal_degree, 'avg_embeddedness': evaluation.avg_embeddedness,
                            'avg_transitivity': evaluation.avg_transitivity, 'conductance': evaluation.conductance,
                            'internal_edge_density': evaluation.internal_edge_density, 'scaled_density': evaluation.scaled_density,
                            'surprise': evaluation.surprise, 'size': evaluation.size, 'expansion': evaluation.expansion}  # 'significance': evaluation.significance

        # Load pandas DataFrame
        if path:
            self.df = pd.read_parquet(path)
            self.name_identifier = os.path.basename(path).split('.')[0]  # Splite filename from base path

        else:
            # No path provided -> choose a random file in the directory corresponding to name
            dirs = os.listdir(f'./data_eval_split/{self.name}')
            file = random.choice(dirs)
            self.df = pd.read_parquet(f'./data_eval_split/{self.name}/{file}')
            self.name_identifier = file.split('.')[0]
        
        # Rescale time if necessary
        self.new_time = new_time
        if new_time:
            self.df = downscale_time(self.df, new_time)  # rescale time 
        else:
            self.new_time = 1

        # Setting minimum distance for contact to be counted
        self.contact_dist_min, self.contact_dist_max = 0.0, 1.5
        self.fov = 2*np.pi / 3 

        # Get trajectories
        self.eval_df_to_trajectory(switch_off_time)

    #@cn.silent_print
    def to_tacoma_tn(self):
        tn = tc.edge_lists()
        tn.N = max(self.df.i.max(), self.df.j.max()) + 1
        Nt = int(60**2 * 24 / self.TU)
        tn.t = list(range(Nt))
        tn.tmax = Nt + 1
        tn.time_unit = '20s'

        contacts = [[] for _ in range(Nt)]

        for _, contact in self.df.iterrows():
            contacts[contact.t + self.temporal_offset].append([contact.i, contact.j])
        
        # Check for errors and convert to edge_changes
        tn.edges = contacts
        print('edge list errors: ', tc.verify(tn))

        tn = tc.convert(tn)
        print('edge changes errors: ', tc.verify(tn))
        self.tn = tn
    
    def eval_df_to_trajectory(self, switch_off_time):
        # This method takes data from an empirically observed network and returns agent trajectories
        # A trajectory starts when an agent has his first edge and ends with its last active edge whenever an agend has no contacts for switch_off_time

        if self.name == 'supermarked':
            self.df_trajectories = pd.read_parquet(f'./data_eval_split/supermarked/{self.name_identifier}_trajectories.parquet')
            self.df_trajectories[['activity_start_min', 'activity_end_min']] = np.floor(self.df_trajectories[['activity_start_min', 'activity_end_min']] / self.new_time).astype('int')
            return self.df_trajectories
        
        else:  
            ij = np.hstack((self.df.i.values, self.df.j.values))
            tt = np.hstack((self.df.t.values, self.df.t.values))
            df2 = pd.DataFrame(data={'ij': ij, 'tt': tt})

            # Group by the 'i' column and aggregate the 't' values into a sorted list
            contacts = df2.groupby('ij')['tt'].apply(lambda x: np.array(sorted(x))).reset_index()
            p_id, activity_start_min, activity_end_min = [], [], []

            for _, person_contact in contacts.iterrows():
                switch_off_points = np.where(np.diff(person_contact.tt) >= switch_off_time)[0] 
                switch_off_points = np.insert(switch_off_points, [0, len(switch_off_points)], [-1, len(person_contact.tt) - 1])
                #print(switch_off_points)

                # Generate trajectories
                for _, (sonp, sofp) in enumerate(zip(switch_off_points[:-1], switch_off_points[1:])):
                    p_id.append(person_contact.ij)
                    #print(person_contact.tt[sonp + 1], person_contact.tt[sofp])
                    activity_start_min.append(person_contact.tt[sonp + 1])
                    activity_end_min.append(person_contact.tt[sofp])
                
                #print(list(zip(switch_off_points[:-1], switch_off_points[1:])))
                #print('\n')
            self.df_trajectories = pd.DataFrame({'p_id': p_id, 'activity_start_min': activity_start_min, 'activity_end_min': activity_end_min})
            return self.df_trajectories
    
    #@cn.silent_print
    def cn_approximation(self, Loc, method, model_kwargs=None, export=False):
        self.Loc = Loc
        if method in ['baseline', 'random', 'clique', 'clique_with_random']:
            sim_TU = 1
        else:
            sim_TU = self.TU

        self.method = method
        ts, te = self.df_trajectories.activity_start_min.min(), self.df_trajectories.activity_end_min.max() 
        CN = cn.ContactNetwork(self.df_trajectories.copy(), Loc, ts, te, sim_TU)
        CN.change_fov(self.fov)

        # Set parameters of model
        if model_kwargs:
            for par, value in model_kwargs.items():
                setattr(CN, par, value)

        CN.make_movement(method=method)
        #print('finished simulation \nstart working on network')

        self.tn_approx = CN.make_tacoma_network(self.contact_dist_min, self.contact_dist_max, sim_TU, export, self.temporal_offset)
        self.paras = f''
        for key, val in CN.paras.items():
            self.paras += f'{key}={val}_'

    def SIR_comparison(self, ax, beta, gamma, nruns, ndays, path, SIR_save_path=None):
        # path = './results/office_empirical_beta=0.013_gamma=3.306878306878307e-05.npy'
        Is = cn.run_SIR(self.tn_approx, self.Loc, self.method, nruns, beta, gamma, ndays, normalize=False, save=False, plot=False)
        time = np.arange(self.tn_approx.tmax * ndays + 1) * 20/(24*60*60)
        #Is = Is[:, :-1]
        #time = time[:-1]
        I_mean, I_err_upper, I_err_lower = util.mean_with_errors(Is, self.tn_approx.N)
        
        # Plot empirical results
        I_emp = np.load(path)
        #I_emp = I_emp[:, :-101]
        I_mean_emp, I_err_upper_emp, I_err_lower_emp = util.mean_with_errors(I_emp, self.tn_approx.N)
        
        # Compute simularity
        time_diff = round((time[np.argmax(I_mean_emp)] - time[np.argmax(I_mean)])/time[np.argmax(I_mean_emp)], 2)
        size_diff = round((np.max(I_mean_emp) - np.max(I_mean)) / self.tn_approx.N, 2)

        # Plot Model results
        if ax:
            ax.plot(time, I_mean / self.tn_approx.N, label='model', c='#ff7f0e')
            ax.fill_between(time, I_err_lower / self.tn_approx.N, I_err_upper / self.tn_approx.N, alpha=.3, color='#ff7f0e')
            ax.set(xlabel='t [d]', ylabel=r'$\sigma$')

            tail = None
            ax.plot(time, I_mean_emp[:] / self.tn_approx.N, label='data', c='#1f77b4')
            ax.fill_between(time, I_err_lower_emp[:] / self.tn_approx.N, I_err_upper_emp[:] / self.tn_approx.N, alpha=.3, color='#1f77b4')

            title = r'$(\arg\max (I_{emp}) - \arg\max (I_{model}))/ \arg\max (I_{emp}) =$' + str(time_diff) + r', $\max (I_{emp}) - \max (I_{model}) =$' + str(size_diff)
            ax.set_title(title)
            ax.legend()
        
        if SIR_save_path:
            print(SIR_save_path)
            print(np.max(I_mean), np.max(I_mean_emp), self.tn_approx.N)
            np.save(SIR_save_path, I_mean)

        return time_diff, size_diff
    
    def SIR_plot_mult(self, ax, beta, gamma, nruns, ndays, tn, label, color, linestyle, ph_emp=None, pt_emp=None):
        # Run SIR
        Is = cn.run_SIR(tn, self.Loc, self.method, nruns, beta, gamma, ndays, normalize=False, save=False, plot=False)
        time = np.arange(tn.tmax * ndays + 1) * 20/(24*60*60)
        Is = Is[:, :-1]
        time = time[:-1]
        I_mean, I_err_upper, I_err_lower = util.mean_with_errors(Is, tn.N)

        # Compute peak heigth and time
        ph = np.max(I_mean)
        pt = time[np.argmax(I_mean)]

        # Plot SIR
        if ph_emp:
            time_diff = str(round((pt_emp - pt) / pt_emp, 2))
            size_diff = str(round((ph_emp - ph) / tn.N, 2))
            label = label + r', $\Delta T$ = ' + time_diff + r', $\Delta \sigma=$' + size_diff

        ax.plot(time, I_mean / tn.N, label=label, c=color, linestyle=linestyle, alpha=.3)
        # ax.fill_between(time, I_err_lower / tn.N, I_err_upper / tn.N, alpha=.3, color=color)
        ax.set(xlabel='t [d]', ylabel=r'$\bar{I}(t)$')

        return ph, pt


    def make_dynetx(self, tn, obs):
        # Convert tacoma dynamic graph to dynetx and list of nx graphs
        '''time = np.array(tn.t).astype('int')
        e_in = tn.edges_in
        e_out = tn.edges_out

        time_stamp, action, i, j = [], [], [], []

        for t, ei, eo in zip(time, e_in, e_out):
            if ei:
                for contact in ei:
                    time_stamp.append(t)
                    i.append(contact[0])
                    j.append(contact[1])
                    action.append('+')
            if eo:        
                for contact in eo:
                    time_stamp.append(t)
                    i.append(contact[0])
                    j.append(contact[1])
                    action.append('-')

        # Create a temporary file to build Dynetx Graph
        df_dynetx = pd.DataFrame({'i': i, 'j': j, 'action': action, 't': time_stamp})
        df_dynetx.to_csv('./temporary_files/edges_in_out_dynetx.tsc', sep='\t', header=False, index=False)

        # Make Dynetx Graph
        g_dynetx = dn.read_interactions('tiles.tsc', nodetype=int, timestamptype=int)'''
        g_dynetx = None

        # Make list of nx Graphs
        tn = tc.convert(tn)

        current_edges = []
        nx_graphs = {}
        for t in range(len(tn.t)):
            current_edges.extend(tn.edges[t])
            if (t + 1) % obs == 0:
                edge_counter = Counter(current_edges)
                edge_list = [(edge[0], edge[1], weight) for edge, weight in edge_counter.items()]
                G = nx.Graph()
                G.add_nodes_from(range(tn.N))
                G.add_weighted_edges_from(edge_list)
                nx_graphs[t] = G
                current_edges = []

        return g_dynetx, nx_graphs
    
    def to_teneto(self):
        pass
            
    def detect_communities(self, temporal_network, method=algorithms.louvain, kwargs={}):
        # Load nx graphs
        _, nx_graphs = self.make_dynetx(temporal_network, obs=1)
        nx_graphs = list(nx_graphs.values())

        # Remove all nodes from snapshots that have less than one edge at given time
        nontrivial_subgraphs = []
        empty_graph_at_t = []
        time_steps = np.arange(0, len(nx_graphs), 1)

        for t, graph in enumerate(nx_graphs):
            # Get all nodes with at least one edge
            degree = np.array(graph.degree)
            nodes_to_plot = degree[:, 0][degree[:, 1] > 0]

            # Check if subgraph is empty
            if len(nodes_to_plot) == 0:
                empty_graph_at_t.append(t)
                continue

            subgraph = graph.subgraph(nodes_to_plot)
            nontrivial_subgraphs.append(subgraph)
        
        # Remove time steps where Graph is empty
        time_steps = np.delete(time_steps, empty_graph_at_t)

        # Create and compute TemporalClustering object
        TC = TemporalClustering()
        for t, graph in enumerate(nontrivial_subgraphs):
            coms = method(graph, **kwargs) 
            TC.add_clustering(coms, t)
        
        self.community_method_parameters = TC.clusterings[0].method_parameters
        return TC, nontrivial_subgraphs

    def cluster_analysis(self, TC, graphs, method, mean):
        observable = []
        func = self.method_func[method]
        for t, graph in zip(TC.get_observation_ids(), graphs):
            coms = TC.get_clustering_at(t)

            if mean:
                observable.append(func(graph, coms, summary=True).score)
            else:
                observable.extend(func(graph, coms, summary=False))
        
        # Drop None values
        observable = [i for i in observable if i is not None]

        return observable

    def plot_cluster_stability(self, TC, ax, smoothing=3, c='#1f77b4', label=None):
        # Get cluster stability trend
        cst = np.array(TC.clustering_stability_trend(evaluation.nf1))
        cst_smooth = util.moving_average(cst, smoothing)

        # Get time
        time = TC.get_observation_ids()

        t_smooth = time[smoothing:]

        # Plot
        ax.plot(time[:-1], cst, label=label + f' raw, mean={round(np.mean(cst), 2)}', c=c, alpha=.3)
        ax.plot(t_smooth, cst_smooth, label=label + f' smoothed {smoothing} TU', c=c)
        ax.legend()
        # ax.set(ylim = (0, 1), xlabel='t', ylabel='community stability', title='NF1 score of time adjacent clusterings')

        return cst, np.mean(cst)

    def plot_cluster_analysis(self, TC, graphs, method, ax, nbins=10, scale='loglog', label=None, color=None, approx=False):
        # Get observation values
        observation_values = np.array(self.cluster_analysis(TC, graphs, method, mean=False))

        # Drop NaNs
        observation_values = observation_values[~np.isnan(observation_values)]


        # Handle axis scale
        xscale, yscale = 'linear', 'linear'
        bins = nbins

        if scale == 'log':
            yscale = 'log'

        elif scale == 'loglog':
            xscale, yscale = 'log', 'log'
            observation_values = observation_values[observation_values > 0]
            # Generate logarithmically spaced bin edges
            bins = np.logspace(np.log10(min(observation_values)), np.log10(max(observation_values)), nbins)

        # Plot histogram
        hist, bins = np.histogram(observation_values, bins, density=True)
        ax.scatter(bins[1:], hist, alpha=.7, color=color, label=label)
        ax.set(xscale=xscale, yscale=yscale, xlabel='value', ylabel='PDF', title=f'{method}')
        if method == 'avg_transitivity':
            ax.set_ylim(0, 1)
        if method == 'avg_embeddedness':
            ax.set_ylim(0, .1)
        ax.legend()

        return observation_values

    def cost_function(self, approx, smoothing, SIR_paras, SIR_path, SIR_save_path, weight=None, plot=False):
        if approx:
            networks = [self.tn, self.tn_approx]
        else:
            networks = [self.tn]
        
        if weight:
            edge_weight = weight
        else:
            edge_weight = 1
        
        CDs, edge_counts = [], []
        eval_results = {}

        for i, tn in enumerate(networks):
            # Contact durations
            res = tc.api.measure_group_sizes_and_durations(tn)
            CDs.append(res.contact_durations)

            # Edge counts
            _, _, m = tc.edge_counts(tn)
            if i > 0:
                m = edge_weight * np.array(m)
            edge_counts.append(m)
            m = moving_average(m[:-1], smoothing)

        try:
            ks_test_CDs = ks_2samp(CDs[0], CDs[1], alternative='two-sided')
            cd_title = 'contact duration distribution, KS: {statistic: .2f} p={pval: .2g}'
            eval_results = util.add_ks_test_results(eval_results, 'contact_duration', CDs[0], CDs[1])
        except ValueError:
            print('CD estimation failed')
            eval_results['contact_duration'] = 1.0
            
        edge_counts_og, edge_counts_appprox = np.array(edge_counts[0]), np.array(edge_counts[1])
        N_e_diff =round((np.sum(edge_counts_og) - np.sum(edge_counts_appprox))/ np.sum(edge_counts_og), 2)
        eval_results['Ne_diff'] = N_e_diff
        
        # Plot SIR results
        beta, gamma, nruns, ndays = SIR_paras
        beta = beta * edge_weight
        time_diff, size_diff = self.SIR_comparison(None, beta=beta, gamma=gamma, nruns=nruns, ndays=ndays, path=SIR_path, SIR_save_path=SIR_save_path)
        eval_results['infection_peak_size_diff'] = size_diff
        eval_results['infection_peak_time_diff'] = time_diff

        #print(eval_results)

        return abs(5*size_diff) + abs(3*time_diff) + abs(2*N_e_diff) + abs(eval_results['contact_duration'])


    def overview_plots(self, approx, ind=None, smoothing=60, weight=None, ntype=None):
        if approx:
            networks = [self.tn, self.tn_approx]
        else:
            networks = [self.tn]
        
        if weight:
            edge_weight = weight
        else:
            edge_weight = 1
        
        labels = [['contact', 'inter contact'], ['model contact', 'model inter contact']]
        colors = ['#1f77b4', '#ff7f0e']
        labels2 = ['data', 'model']
        ICTs, CDs, degrees, edge_counts = [], [], [], []
        eval_results = {}

        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        (ax1, ax2, ax3, ax4) = axs.flatten()
        for i, tn in enumerate(networks):
            # Contact durations
            res = tc.api.measure_group_sizes_and_durations(tn)
            CDs.append(res.contact_durations)
            plot_contact_durations(res, (ax1, ax3), fit_power_law=True, bins=50, xlabel='duration [min]', color=colors[i], label=labels[i])
            ax3.clear()

            # Edge counts
            _, _, m = tc.edge_counts(tn)
            if i > 0:
                m = edge_weight * np.array(m)
            edge_counts.append(m)
            m, t = moving_average(m[:-1], smoothing), moving_average(tn.t, smoothing)
            ax2.plot(t * 20/(24*60**2), m, color=colors[i], label=labels2[i], alpha=.5)

        try:
            ks_test_CDs = ks_2samp(CDs[0], CDs[1], alternative='two-sided')
            cd_title = 'contact duration distribution, KS: {statistic: .2f} p={pval: .2g}'
            ax1.set_title(cd_title.format(statistic=ks_test_CDs.statistic, pval=ks_test_CDs.pvalue))
            eval_results = util.add_ks_test_results(eval_results, 'contact_duration', CDs[0], CDs[1])
        except ValueError:
            print('CD estimation failed')
            

        # handle different time handling between approaches
        '''if len(edge_counts[0]) > len(edge_counts[1]):
            edge_counts_og, edge_counts_appprox = np.array(edge_counts[0][:-1]), np.array(edge_counts[1])
        elif len(edge_counts[0]) < len(edge_counts[1]):
            edge_counts_og, edge_counts_appprox = np.array(edge_counts[0]), np.array(edge_counts[1][:-1])
        else:
            edge_counts_og, edge_counts_appprox = np.array(edge_counts[0]), np.array(edge_counts[1])'''
        
        edge_counts_og, edge_counts_appprox = np.array(edge_counts[0]), np.array(edge_counts[1])
        N_e_diff =round((np.sum(edge_counts_og) - np.sum(edge_counts_appprox))/ np.sum(edge_counts_og), 2)
        eval_results['Ne_diff'] = N_e_diff
        # mean_absolute_error = np.mean(np.absolute(edge_counts_og - edge_counts_appprox))
        edge_title = f'rolling average (smoothing={smoothing}TU), ' + r'$(N_E-N_{E,model})/N_E=$' + f'{N_e_diff}' 
        ax2.set_title(edge_title)

        for ax in axs.flatten():
            ax.legend()
        
        # Plot SIR results
        if ntype == 'office':
            SIR_path = './results/office_empirical_beta=0.013_gamma=3.306878306878307e-05.npy'
            beta = .013
        elif ntype == 'primaryschool':
            SIR_path = './results/primaryschool_empirical_beta=0.0013_gamma=3.306878306878307e-05.npy'
            beta = .0013
        elif ntype == 'highschool':
            beta = .007
            SIR_path = './results/highschool_empirical_beta=0.007_gamma=3.306878306878307e-05.npy'
        elif ntype == 'supermarked':
            beta = .075
            SIR_path = './results/supermarked_empirical_beta=0.075_gamma=3.306878306878307e-05.npy'
        else:
            beta = .007
            SIR_path = './results/highschool_empirical_beta=0.007_gamma=3.306878306878307e-05.npy'
        
        # './results/primaryschool_empirical_beta=0.0013_gamma=3.306878306878307e-05.npy' # primary 0.0013
        # './results/office_empirical_beta=0.013_gamma=3.306878306878307e-05.npy' # office 0.013
        gamma, nruns, ndays = 20 / (7*24*60*60), 100, 35
        time_diff, size_diff = self.SIR_comparison(ax3, beta=edge_weight * beta, gamma=gamma, nruns=nruns, ndays=ndays, path=SIR_path)
        eval_results['infection_peak_size_diff'] = size_diff
        eval_results['infection_peak_time_diff'] = time_diff


        cd_diff = eval_results['contact_duration']
        fig.suptitle(f'{self.name}, A/N_v = {self.Loc.loc_id}, {self.method}, cost: {round(abs(5*size_diff) + abs(3*time_diff) + abs(2*N_e_diff) + abs(cd_diff), 2)}')
        plt.tight_layout()
        #plt.savefig(f'./plots/eval_networks/overview_highschool_f0_2013-12-03_{self.method}/loc_id={self.Loc.loc_id}/{self.paras[:-1]}.png')
        plt.savefig(f'{self.name}_best_para_run_{self.method}_loc_id={self.Loc.loc_id}')
        plt.close()

        return round(abs(5*size_diff) + abs(3*time_diff) + abs(2*N_e_diff) + abs(cd_diff), 2)

    def overview_mult(self, method, model_kwargs, school_space, Npps, nruns=5, smoothing=3, networks=None):
        Loc_hs = cn.build_location(self.df_trajectories.copy(), loc_id=school_space, loc_type='school', N_pps=Npps, school_space=school_space)
        self.Loc = Loc_hs
        self.method = method
        cost_per_run = np.zeros(nruns)

        if not networks:
            networks = []

            if nruns % 2 == 0:
                nruns += 1
                print('nruns must be odd, I added +1 for you')
            
            # Build and evaluate networks
            for i in range(nruns):
                print(i)
                self.cn_approximation(Loc_hs, method, model_kwargs=model_kwargs, export=False)
                networks.append(self.tn_approx)

                cost_per_run[i] = self.cost_function(True, smoothing=3)


            # Select median, best, worst network
            networks = np.array(networks)
            networks = networks[np.argsort(cost_per_run)]
            median_network = networks[int(nruns / 2)]
            best_network = networks[0]
            worst_network = networks[-1]
        
        else:
            median_network, best_network, worst_network = networks

        # Plot empirical, median, best, worst
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.name}, {method}')
        (ax1, ax2, ax3, ax4) = axs.flatten()

        # Empirical
        # Contact duration
        res = tc.api.measure_group_sizes_and_durations(self.tn)
        plot_contact_durations(res, (ax1, ax4), fit_power_law=False, bins=50, xlabel='duration [20sec]', color='#ff7f0e', label=['empirical', 'inter contact'], alpha=.7)
        cd_emp = res.contact_durations
        ax1.set_title('contact durations')
        # Edge counts
        _, _, m_emp = tc.edge_counts(self.tn)
        m_emp, t = moving_average(m_emp[:-1], smoothing), moving_average(self.tn.t, smoothing)
        m_emp = np.array(m_emp)
        ax2.plot(t * 20/(24*60**2), m_emp, color='#ff7f0e', label='empirical', alpha=.7)
        ax2.set(title=f'edge counts smoothed by {smoothing}TU', xlabel=r'$t$[d]', ylabel=r'$N_E(t)$')
        #SIR
        beta, gamma, nruns, ndays = .007, 20 / (7*24*60*60), 100, 50
        ph_emp, pt_emp = self.SIR_plot_mult(ax3, beta, gamma, nruns, ndays, self.tn, linestyle='-', label='empirical', color='#ff7f0e')

        # Median, best, worst
        for name, network, alpha, marker, linestyle in zip(['median', 'best', 'worst'], [median_network, best_network, worst_network], [.7, .5, .5], ['o', 'x', 'v'], ['-', '--', ':']):
            # Contact duration
            res = tc.api.measure_group_sizes_and_durations(network)
            cd = res.contact_durations
            ks_test_CDs = ks_2samp(cd_emp, cd, alternative='two-sided')
            cd_label = name + ', KS: {statistic: .2f} p={pval: .2g}'
            cd_label = cd_label.format(statistic=ks_test_CDs.statistic, pval=ks_test_CDs.pvalue)
            plot_contact_durations(res, (ax1, ax4), fit_power_law=False, bins=50, xlabel='duration [min]', color='#1f77b4', label=[cd_label, 'inter contact'], alpha=alpha, marker=marker)
            ax4.clear()
            
            # Edge counts
            _, _, m = tc.edge_counts(network)
            m, t = moving_average(m[:-1], smoothing), moving_average(network.t, smoothing)
            m = np.array(m)
            N_e_diff = round((np.sum(m_emp) - np.sum(m)) / np.sum(m_emp), 2)
            label = f'{name}, '+ r'$N_{E,\, diff}=$' + str(N_e_diff)
            ax2.plot(t * 20/(24*60**2), m, color='#1f77b4', label=label, alpha=alpha, linestyle=linestyle)

            # SIR
            _ = self.SIR_plot_mult(ax3, beta, gamma, nruns, ndays, network, linestyle=linestyle, label=name, color='#1f77b4', ph_emp=ph_emp, pt_emp=pt_emp)

        # Scores
            ax4.hist(cost_per_run, bins=10, density=True)
            ax4.set(xlabel='score',ylabel='PDF')
            ax4.set_title('Scores PDF')
        for ax in axs.flatten():
            ax.legend()

        plt.tight_layout()
        plt.savefig(f'{school_space}_{method}_plot.png')

        tc.write_json_taco(best_network, f'{school_space}_{method}_best.taco')
        tc.write_json_taco(worst_network, f'{school_space}_{method}_worst.taco')
        tc.write_json_taco(median_network, f'{school_space}_{method}_median.taco')


class SuperTemporalClustering(TemporalClustering):
    @staticmethod
    def jaccard(x, y):
        return len(set(x) & set(y)) / len(set(x) | set(y))
    
    # Wrapper for temporal clustering that holds network snapshots as well as some often repeated code
    def __init__(self, EN_list):
        # Init parentclass
        super().__init__()

        # Available methods for evaluation
        self.method_func = {'avg_internal_degree': evaluation.average_internal_degree, 'avg_embeddedness': evaluation.avg_embeddedness,
                            'avg_transitivity': evaluation.avg_transitivity, 'conductance': evaluation.conductance,
                            'internal_edge_density': evaluation.internal_edge_density, 'scaled_density': evaluation.scaled_density,
                            'surprise': evaluation.surprise, 'size': evaluation.size}  # 'significance': evaluation.significance

        # Extract networks from EvaluationNetworks
        nx_graphs_all_days = []
        self.EN_list = EN_list
        for EN in self.EN_list:
            EN.to_tacoma_tn(20, 1000)
            obs = 1
            _, nx_graphs_single = EN.make_dynetx(EN.tn, obs=obs)
            nx_graphs_all_days.append(nx_graphs_single)
        
        all_graphs_aggregated = []
        for nx_graphs_single in nx_graphs_all_days:
            all_graphs_aggregated.extend(list(nx_graphs_single.values()))

        self.nx_graphs = all_graphs_aggregated

        # Remove all nodes from snapshots that have less than one edge at given time
        self.nontrivial_subgraphs = []
        self.empty_graph_at_t = []
        self.time_steps = np.arange(0, len(self.nx_graphs), 1)

        for t, graph in enumerate(self.nx_graphs):
            # Get all nodes with at least one edge
            degree = np.array(graph.degree)
            nodes_to_plot = degree[:, 0][degree[:, 1] > 0]

            # Check if subgraph is empty
            if len(nodes_to_plot) == 0:
                self.empty_graph_at_t.append(t)
                continue

            subgraph = graph.subgraph(nodes_to_plot)
            self.nontrivial_subgraphs.append(subgraph)
        
        # Remove time steps where Graph is empty
        self.time_steps = np.delete(self.time_steps, self.empty_graph_at_t)
    
    def approximate_network(self, method, Loc, model_kwargs, TU, overview=False):
        self.ApproxTC = TemporalClustering()

        # Approximate observed network with given method and parameters
        nx_graphs_all_days = []
        for EN in self.EN_list:
            EN.cn_approximation(Loc, method, TU, model_kwargs=model_kwargs, export=False, temporal_offset=None)
            obs=1
            _, nx_graphs_single = EN.make_dynetx(EN.tn_approx, obs=obs)
            EN.overview_plots(True, 0, 3)
            nx_graphs_all_days.append(nx_graphs_single)
        
        all_graphs_aggregated = []
        for nx_graphs_single in nx_graphs_all_days:
            all_graphs_aggregated.extend(list(nx_graphs_single.values()))

        self.nx_graphs_approx = all_graphs_aggregated

        # Remove all nodes from snapshots that have less than one edge at given time
        self.nontrivial_subgraphs_approx = []
        empty_graph_at_t = []
        self.time_steps_approx = np.arange(0, len(self.nx_graphs_approx), 1)

        for t, graph in enumerate(self.nx_graphs_approx):
            # Get all nodes with at least one edge
            degree = np.array(graph.degree)
            nodes_to_plot = degree[:, 0][degree[:, 1] > 0]

            # Check if subgraph is empty
            if len(nodes_to_plot) == 0:
                empty_graph_at_t.append(t)
                continue

            subgraph = graph.subgraph(nodes_to_plot)
            self.nontrivial_subgraphs_approx.append(subgraph)
        
        # Remove time steps where Graph is empty
        self.time_steps_approx = np.delete(self.time_steps_approx, empty_graph_at_t)

    
    def detect_communities(self, method=algorithms.louvain, kwargs={}, approx=False):
        # Check wether communities in approximated network or in original network should be detected
        if approx:
            subgraphs = self.nontrivial_subgraphs_approx
            Parent = self.ApproxTC
        else:
            subgraphs = self.nontrivial_subgraphs
            Parent = super()

        for t, graph in enumerate(subgraphs):
            coms = method(graph, **kwargs) 
            Parent.add_clustering(coms, t)
        
        self.method_parameters = self.clusterings[0].method_parameters

    def calculate_matches(self, method=None, two_sided=False):
        if not method:
            method = self.jaccard
        
        self.matches = super().community_matching(method, two_sided=two_sided)

    def cluster_analysis(self, method, mean, approx):
        # Check wether communities in approximated network or in original network should be detected
        if approx:
            subgraphs = self.nontrivial_subgraphs_approx
            Parent = self.ApproxTC
        else:
            subgraphs = self.nontrivial_subgraphs
            Parent = super()

        observable = []
        func = self.method_func[method]
        for t, graph in zip(Parent.get_observation_ids(), subgraphs):
            coms = Parent.get_clustering_at(t)

            if mean:
                observable.append(func(graph, coms, summary=True).score)
            else:
                observable.extend(func(graph, coms, summary=False))
        
        # Drop None values
        observable = [i for i in observable if i is not None]

        return observable

    def plot_cluster_analysis(self, method, ax, nbins=10, scale='linear', color=None, approx=False):
        if approx:
            label = 'model'
        else:
            label = 'data'
        # Get observation values
        observation_values = np.array(self.cluster_analysis(method, mean=False, approx=approx))

        # Drop NaNs
        observation_values = observation_values[~np.isnan(observation_values)]


        # Handle axis scale
        xscale, yscale = 'linear', 'linear'
        bins = nbins

        if scale == 'log':
            yscale = 'log'

        elif scale == 'loglog':
            xscale, yscale = 'log', 'log'
            observation_values = observation_values[observation_values > 0]
            # Generate logarithmically spaced bin edges
            bins = np.logspace(np.log10(min(observation_values)), np.log10(max(observation_values)), nbins)

        # Plot histogram
        hist, bins = np.histogram(observation_values, bins, density=True)
        ax.scatter(bins[1:], hist, alpha=.7, color=color, label=label)
        ax.set(xscale=xscale, yscale=yscale, xlabel='value', ylabel='probability', title=f'{method}')

    def plot_cluster_timeseries(self, method, ax, smoothing=10, color='#1f77b4'):
        # Get timeseries
        timeseries = np.array(self.cluster_analysis(method, mean=True))
        timeseries_smooth = util.moving_average(timeseries, smoothing)
        t_smooth = self.time_steps[smoothing - 1:]

        # Plot
        ax.plot(self.time_steps, timeseries, label=f'raw mean={round(np.mean(timeseries), 2)}')
        ax.plot(t_smooth, timeseries_smooth, label=f'smooth {smoothing}TU')
        ax.set(xlabel='t', ylabel=f'avg per TU of {method}', title=method)
        ax.legend()

    def plot_cluster_timeseries_overview(self):
        fig, axs = plt.subplots(3, 3, figsize=(15, 12))
        axs = axs.flatten()

        for method, ax in zip(self.method_func.keys(), axs):
            self.plot_cluster_timeseries(method, ax)
            
        plt.tight_layout()

    def plot_cluster_analysis_overview(self, scale, figaxs, approx=False):
        if figaxs:
            fig, axs = figaxs
        else:
            fig, axs = plt.subplots(3, 3, figsize=(15, 12))
            axs = axs.flatten()

        for method, ax in zip(self.method_func.keys(), axs):
            self.plot_cluster_analysis(method, ax, scale=scale, approx=approx)
            ax.legend()
            
        plt.tight_layout()

    def plot_communities(self, t0, nrwos=3, ncols=3):
        nplots = nrwos * ncols
        fig, axs = plt.subplots(nrwos, ncols, figsize=(15, 15))
        axs = axs.flatten()
        pos = None
        communities = {}

        for t, ax in zip(range(t0, t0 + nplots), axs):
            # Get communities/graph at time t
            coms = super().get_clustering_at(t)
            graph = self.nontrivial_subgraphs[t]
            # Assign community to node
            for com_id, com in enumerate(coms.communities):
                for node in com:
                    communities[node] = com_id
            
            colors = []
            for node in graph.nodes():
                colors.append(communities[node])

            ax.set(title=f't={t}')
            pos = nx.spring_layout(graph, pos=pos, k=1)
            pc = nx.draw_networkx_nodes(graph, pos=pos, node_color=colors, ax=ax, node_size=100, cmap='jet', alpha=.5)
            nx.draw_networkx_edges(graph, pos=pos, ax=ax, alpha=.5)
            nx.draw_networkx_labels(graph, pos=pos, ax=ax)
            fig.colorbar(pc)
        
        plt.tight_layout()
    
    def plot_community_stability_trend(self, smoothing=10, approx=False, figaxs=None):
        if approx:
            subgraphs = self.nontrivial_subgraphs_approx
            Parent = self.ApproxTC
            time = self.time_steps_approx
            print(len(subgraphs), len(time))
        else:
            subgraphs = self.nontrivial_subgraphs
            Parent = super()
            time = self.time_steps

        if figaxs:
            fig, ax = figaxs
        else:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Get cluster stability trend
        cst = np.array(Parent.clustering_stability_trend(evaluation.nf1))
        print(len(cst))
        cst_smooth = util.moving_average(cst, smoothing)
        print(len(cst_smooth))

        # Get number of edges/nodes per time step
        nedges, nnodes = np.zeros_like(time[:-1]), np.zeros_like(time[:-1])
        for t, graph in enumerate(subgraphs[:-1]):
            nedges[t] = graph.number_of_edges()
            nnodes[t] = graph.number_of_nodes()
            
        # Normalize nedges, nnodes and smooth
        nedges = nedges / np.max(nedges)
        nedges = util.moving_average(nedges, smoothing)
        nnodes = nnodes / np.max(nnodes)
        nnodes = util.moving_average(nnodes, smoothing)

        # Compute correlation
        print(len(cst_smooth), len(nnodes))
        nodes_pearson_res = pearsonr(cst_smooth, nnodes)
        nodes_r, nodes_p = round(nodes_pearson_res.statistic, 2), round(nodes_pearson_res.pvalue, 2)
        edges_pearson_res = pearsonr(cst_smooth, nedges)
        edges_r, edges_p = round(edges_pearson_res.statistic, 2), round(edges_pearson_res.pvalue, 2)

        # Get smoothed time
        t_smooth = time[smoothing:]

        # Plot
        ax.plot(time[:-1], cst, label=f'raw, mean={round(np.mean(cst), 2)}')
        ax.plot(t_smooth, cst_smooth, label=f'smoothed {smoothing} TU')
        ax.plot(t_smooth, nedges, label=f'#edges, R={edges_r} p={edges_p}')
        ax.plot(t_smooth, nnodes, label=f'#nodes, R={nodes_r} p={nodes_p}')
        ax.set(ylim = (0, 1), xlabel='t', ylabel='community stability', title='NF1 score of time adjacent clusterings')
        ax.legend()


def model_parameters(trial, method):
    if method == 'clique_with_random':
        Npps =  trial.suggest_int('Npps', 1, 40)
        p_space_change = trial.suggest_float('p_space_change', .00001, .1)
        mean = trial.suggest_int('mean', 5, 720)
        sigma = trial.suggest_int('sigma', 1, 100)
        p_add = trial.suggest_float('p_add', .00001, .1)
        pareto_shape = trial.suggest_float('pareto_shape', .1, 10.)
        model_kwargs = {'p_space_change': p_space_change, 'mean': mean, 'sigma': sigma, 'p_add': p_add, 'pareto_shape': pareto_shape}
    
    elif method == 'baseline':
        Npps = 20
        weight = trial.suggest_float('weight', 0, 0.01)  
        model_kwargs = {'weight': weight}
    
    elif method == 'random':
        Npps = 20
        p_add = trial.suggest_float('p_add', .00001, .01)
        pareto_shape = trial.suggest_float('pareto_shape', .1, 10.)
        model_kwargs = {'p_add': p_add, 'pareto_shape': pareto_shape}

    elif method == 'clique':
        Npps =  trial.suggest_int('Npps', 1, 40)
        p_space_change = trial.suggest_float('p_space_change', .00001, .1)
        mean = trial.suggest_int('mean', 5, 720)
        sigma = trial.suggest_int('sigma', 1, 100)
        model_kwargs = {'p_space_change': p_space_change, 'mean': mean, 'sigma': sigma}
    
    elif method == 'RWP':
        Npps = 20
        rwp_wt_max = trial.suggest_int('rwp_wt_max', 10, 3600)
        model_kwargs = {'RWP_WT_MAX': rwp_wt_max}

    elif method == 'TLW':
        Npps = 20
        TLW_WT_MAX = trial.suggest_int('TLW_WT_MAX', 1, 3600)  # maximum value of the waiting time distribution. Default is 100
        TLW_WT_EXP = trial.suggest_float('TLW_WT_EXP', -10.0, -.1)  # exponent of the waiting time distribution. Default is -1.8
        FL_MAX = trial.suggest_int('FL_MAX', 10, 100)# maximum value of the flight length distribution. Default is 50
        FL_EXP = trial.suggest_float('FL_EXP', -10.0, -.1)  # exponent of the flight length distribution. Default is -2.6
        model_kwargs = {'TLW_WT_MAX': TLW_WT_MAX, 'TLW_WT_EXP': TLW_WT_EXP, 'FL_MAX': FL_MAX, 'FL_EXP': FL_EXP}

    elif method == 'STEPS_with_RWP_pareto':
        Npps = trial.suggest_int('Npps', 1, 40)
        k = trial.suggest_float('k', 1.1, 10.0)
        STEPS_pareto = trial.suggest_float('STEPS_pareto', .1, 3.)
        rwp_wt_max = trial.suggest_int('rwp_wt_max', 10, 3600)
        model_kwargs = {'k': k, 'STEPS_pareto': STEPS_pareto, 'RWP_WT_MAX': rwp_wt_max}
    
    elif method == 'STEPS_with_RWP':
        Npps = trial.suggest_int('Npps', 1, 40)
        k = trial.suggest_float('k', 1.1, 3.0)
        STEPS_pause_min = trial.suggest_int('STEPS_pause_min', 10, 3600)
        STEPS_pause = trial.suggest_int('STEPS_pause', 10, 3600)
        rwp_wt_max = trial.suggest_int('rwp_wt_max', 10, 3600)
        STEPS_pause_max = STEPS_pause_min + STEPS_pause
        model_kwargs = {'k': k, 'STEPS_pause_min': STEPS_pause_min, 'STEPS_pause_max': STEPS_pause_max, 'RWP_WT_MAX': rwp_wt_max}
        
    elif method == 'STEPS_pareto':
        Npps = trial.suggest_int('Npps', 1, 40)
        k = trial.suggest_float('k', 1.1, 10.0)
        STEPS_pareto = trial.suggest_float('STEPS_pareto', .1, 3.)
        model_kwargs = {'k': k, 'STEPS_pareto': STEPS_pareto}

    elif method == 'STEPS':
        Npps = trial.suggest_int('Npps', 1, 40)
        k = trial.suggest_float('k', 1.1, 3.0)
        STEPS_pause_min = trial.suggest_int('STEPS_pause_min', 10, 3600)
        STEPS_pause = trial.suggest_int('STEPS_pause', 10, 3600)
        STEPS_pause_max = STEPS_pause_min + STEPS_pause
        model_kwargs = {'k': k, 'STEPS_pause_min': STEPS_pause_min, 'STEPS_pause_max': STEPS_pause_max}
    return Npps, model_kwargs


def hyperparameter_tuning(sp, SW, ntype, method):
    sot = 2000
    new_time = None

    if ntype == 'office':
        capacity = 217
        path = './data_eval_split/InVS/f1_1970-01-01.parquet' 
        max_dist, min_dist, fov = 1.5, 0.0, 2.094395
        SIR_paras = .013, 20 / (7*24*60*60), 250, 35  # beta, gamma, nruns, ndays
        SIR_path = './results/office_empirical_beta=0.013_gamma=3.306878306878307e-05.npy'

    elif ntype == 'highschool':
        capacity = 327
        path = './data_eval_split/highschool/f0_2013-12-03.parquet'
        max_dist, min_dist, fov = 1.5, 0.0, 2.094395
        SIR_paras = .007, 20 / (7*24*60*60), 250, 35
        SIR_path = './results/highschool_empirical_beta=0.007_gamma=3.306878306878307e-05.npy'

    elif ntype == 'primaryschool':
        capacity = 242
        path = './data_eval_split/primaryschool/1970-01-01.parquet'
        max_dist, min_dist, fov = 1.5, 0.0, 2.094395
        SIR_paras = .0013, 20 / (7*24*60*60), 250, 35
        SIR_path = './results/primaryschool_empirical_beta=0.0013_gamma=3.306878306878307e-05.npy'

    elif ntype == 'supermarked':
        capacity = 112.2
        path = './data_eval_split/supermarked/f0_2021-03-17.parquet'
        max_dist, min_dist, fov = 1.5, 0.0, 2.094395
        new_time = 20
        SIR_paras = .075, 20 / (7*24*60*60), 250, 35
        SIR_path = './results/supermarked_empirical_beta=0.075_gamma=3.306878306878307e-05.npy'
        

    EN = EvaluationNetwork(ntype, TU=20, temporal_offset=1000, path=path, switch_off_time=sot, new_time=new_time)
    EN.fov = fov
    EN.contact_dist_min = min_dist
    EN.contact_dist_max = max_dist
    EN.to_tacoma_tn()
    study = SW.study
    stds = SW.stds

    try:
        best_score = study.best_value
    except ValueError:
        best_score = 10_000

    def model(trial):
        Npps, model_kwargs = model_parameters(trial, method)

        if method == 'baseline':
            nruns = 1
            weight = model_kwargs['weight']
            model_kwargs = {}
        else:
            weight = None
            nruns = 20

        cost_per_run = np.zeros(nruns)
        print(f'Location: {ntype}, Method: {method}, Npps: {Npps}, kwargs: {model_kwargs}')

        for i in range(nruns):
            Loc = cn.build_location(EN.df_trajectories.copy(), loc_id=sp, loc_type=ntype, N_pps=Npps, school_space=sp, capacity=capacity)
            EN.cn_approximation(Loc, method, model_kwargs=model_kwargs, export=False,)
            cost_per_run[i] = EN.cost_function(approx=True, smoothing=3, weight=weight, SIR_paras=SIR_paras, SIR_path=SIR_path, SIR_save_path=None)
            print(i, cost_per_run[i])

            if cost_per_run[i] > 10.0:
                cost_per_run = np.zeros(nruns) + cost_per_run[i]
                break

            if i < nruns - 1:
                if (i >= 4) and (np.mean(cost_per_run[:i + 1]) - best_score) > 1.:
                    cost_per_run = np.zeros(nruns) + np.mean(cost_per_run[:i + 1])
                    break

        print(np.mean(cost_per_run), np.std(cost_per_run), cost_per_run)

        stds.append(np.std(cost_per_run))
        return np.mean(cost_per_run)
    
    study.optimize(model, n_trials=3)
    
    with open(f'{ntype}_tuned_paras_{method}_loc_id={sp}.pkl', 'wb') as file:   
        # A new file will be created 
        pickle.dump(SW, file) 


def generate_best_para_realizations(ntype, method, nruns=21):
    best_paras_office = {'STEPS_pareto': {'k': 2.8809814385623684,
                            'STEPS_pareto': 0.7676296683599062,
                            'N_people_per_space': 24},
                            'STEPS_with_RWP_pareto': {'k': 5.120812726526037,
                            'STEPS_pareto': 0.3463158890907987,
                            'N_people_per_space': 40,
                            'RWP_WT_MAX': 317},
                            'clique_with_random': {'p_space_change': 0.026573954773857322,
                            'mean': 271,
                            'sigma': 28,
                            'p_add': 0.038092639210002804,
                            'pareto_shape': 1.592905968448615,
                            'N_people_per_space': 2},
                            'RWP': {'RWP_WT_MAX': 1797},
                            'random': {'p_add': 0.00045426551451094066, 'pareto_shape': 3.08156770075869},
                            'baseline': {'weight': 0.0005436035657689404},
                            'TLW': {'TLW_WT_MAX': 3277,
                            'TLW_WT_EXP': -0.28634300032673937,
                            'FL_MAX': 49,
                            'FL_EXP': -9.894483858860687}}
    
    best_paras_primaryschool = {'STEPS_pareto': {'k': 9.974003647535408,
                                    'STEPS_pareto': 0.6130675933667971,
                                    'N_people_per_space': 39},
                                    'STEPS_with_RWP_pareto': {'k': 7.870151829133263,
                                    'STEPS_pareto': 0.5202840800155737,
                                    'N_people_per_space': 22,
                                    'RWP_WT_MAX': 3600},
                                    'clique_with_random': {'p_space_change': 0.06605636980052534,
                                    'mean': 514,
                                    'sigma': 33,
                                    'p_add': 0.046199106031901936,
                                    'pareto_shape': 7.2632611210982345,
                                    'N_people_per_space': 9},
                                    'random': {'p_add': 0.0009190357020863638, 'pareto_shape': 2.455624393138444},
                                    'RWP': {'RWP_WT_MAX': 2108},
                                    'baseline': {'weight': 0.0013047375595401833},
                                    'TLW': {'TLW_WT_MAX': 2307,
                                    'TLW_WT_EXP': -0.6106044193554402,
                                    'FL_MAX': 85,
                                    'FL_EXP': -8.996563363396556}}
    
    best_paras_highschool = {'STEPS_pareto': {'k': 4.3879605793130265,
                                'STEPS_pareto': 0.4210183226476539,
                                'N_people_per_space': 27},
                                'clique_with_random': {'p_space_change': 0.0562632459155923,
                                'mean': 170,
                                'sigma': 91,
                                'p_add': 0.008016298535436495,
                                'pareto_shape': 9.998054452780455,
                                'N_people_per_space': 19},
                                'random': {'p_add': 0.00034618042207039445,
                                'pareto_shape': 7.061619041378239},
                                'baseline': {'weight': 0.0002992849250132354},
                                'STEPS_with_RWP_pareto': {'k': 8.128418593329403,
                                'STEPS_pareto': 0.11968494818705015,
                                'N_people_per_space': 21,
                                'RWP_WT_MAX': 3588},
                                'RWP': {'RWP_WT_MAX': 3599},
                                'TLW': {'TLW_WT_MAX': 2359,
                                'TLW_WT_EXP': -0.5029329300743824,
                                'FL_MAX': 80,
                                'FL_EXP': -9.987833730779137}}

    best_paras_supermarked = {'clique_with_random': {'p_space_change': 0.00820944885032916,
                                'mean': 574,
                                'sigma': 43,
                                'p_add': 0.09139312807734347,
                                'pareto_shape': 2.1211790041196585,
                                'N_people_per_space': 31},
                                'random': {'p_add': 0.009703836159227257, 'pareto_shape': 1.4881299058280244},
                                'STEPS_with_RWP_pareto': {'k': 1.1421931631671745,
                                'STEPS_pareto': 2.0289036323671814,
                                'N_people_per_space': 33,
                                'RWP_WT_MAX': 388},
                                'STEPS_pareto': {'k': 1.1345087010817283,
                                'STEPS_pareto': 2.4205491317513514,
                                'N_people_per_space': 12},
                                'TLW': {'TLW_WT_MAX': 3258,
                                'TLW_WT_EXP': -5.025126349509932,
                                'FL_MAX': 91,
                                'FL_EXP': -0.11181436854902468},
                                'RWP': {'RWP_WT_MAX': 11},
                                'baseline': {'weight': 0.009970022805909147}}
    
    sot = 2000
    new_time = None

    if ntype == 'office':
        capacity = 217
        path = './data_eval_split/InVS/f1_1970-01-01.parquet' 
        max_dist, min_dist, fov = 1.5, 0.0, 2.094395
        SIR_paras = .013, 20 / (7*24*60*60), 250, 35  # beta, gamma, nruns, ndays
        SIR_path = './results/office_empirical_beta=0.013_gamma=3.306878306878307e-05.npy'
        best_paras = best_paras_office

    elif ntype == 'highschool':
        capacity = 327
        path = './data_eval_split/highschool/f0_2013-12-03.parquet'
        max_dist, min_dist, fov = 1.5, 0.0, 2.094395
        SIR_paras = .007, 20 / (7*24*60*60), 250, 35
        SIR_path = './results/highschool_empirical_beta=0.007_gamma=3.306878306878307e-05.npy'
        best_paras = best_paras_highschool

    elif ntype == 'primaryschool':
        capacity = 242
        path = './data_eval_split/primaryschool/1970-01-01.parquet'
        max_dist, min_dist, fov = 1.5, 0.0, 2.094395
        SIR_paras = .0013, 20 / (7*24*60*60), 250, 35
        SIR_path = './results/primaryschool_empirical_beta=0.0013_gamma=3.306878306878307e-05.npy'
        best_paras = best_paras_primaryschool

    elif ntype == 'supermarked':
        capacity = 112.2
        path = './data_eval_split/supermarked/f0_2021-03-17.parquet'
        max_dist, min_dist, fov = 1.5, 0.0, 2.094395
        new_time = 20
        SIR_paras = .075, 20 / (7*24*60*60), 250, 35
        SIR_path = './results/supermarked_empirical_beta=0.075_gamma=3.306878306878307e-05.npy'
        best_paras = best_paras_supermarked
    
    model_kwargs = best_paras[method]

    if method == 'baseline':
            weight = model_kwargs['weight']
    else:
        weight=None

    try:
        Npps = model_kwargs['N_people_per_space']
        del model_kwargs['N_people_per_space']
    except:
        Npps = 20
    

    EN = EvaluationNetwork(ntype, TU=20, temporal_offset=1000, path=path, switch_off_time=sot, new_time=new_time)
    EN.fov = fov
    EN.contact_dist_min = min_dist
    EN.contact_dist_max = max_dist
    EN.to_tacoma_tn()
    tns = []

    cost_per_run = np.zeros(nruns)
    print(f'Location: {ntype}, Method: {method}, Npps: {Npps}, kwargs: {model_kwargs}')

    for i in range(nruns):
        Loc = cn.build_location(EN.df_trajectories.copy(), loc_id=None, loc_type=ntype, N_pps=Npps, school_space=None, capacity=capacity)
        EN.cn_approximation(Loc, method, model_kwargs=model_kwargs, export=False,)
        tns.append(EN.tn_approx)
        taco_save_path = f'./networks/{ntype}/optimized_networks/{method}/{method}_{str(i).zfill(2)}_network.taco' 
        SIR_save_path = f'./networks/{ntype}/optimized_networks/{method}/{method}_{str(i).zfill(2)}_SIR.npy'
        #taco_save_path = './trash/taco'
        #SIR_save_path = './trash/SIR'
        tc.write_json_taco(EN.tn_approx, taco_save_path)
        cost_per_run[i] = EN.cost_function(approx=True, smoothing=3, weight=weight, SIR_paras=SIR_paras, SIR_path=SIR_path, SIR_save_path=SIR_save_path)
        print(i, cost_per_run[i])

    # Select median network
    sorted_indices = np.argsort(cost_per_run)
    median_index = len(sorted_indices) // 2  # If length is odd, this is the median; if even, this is the lower median index
    median_entry_index = sorted_indices[median_index]
    median_val = cost_per_run[median_entry_index]
    print(median_entry_index, median_val)
    tc.write_json_taco(tns[median_entry_index], f'./networks/{ntype}/medians/{method}_median={median_val}_network.taco')
    shutil.copy(f'./networks/{ntype}/optimized_networks/{method}/{method}_{str(median_entry_index).zfill(2)}_SIR.npy', f'./networks/{ntype}/medians/' )


def best_para_many_runs(ntype, method):    
    best_paras_office = {'STEPS_pareto': {'k': 2.8809814385623684,
                            'STEPS_pareto': 0.7676296683599062,
                            'N_people_per_space': 24},
                            'STEPS_with_RWP_pareto': {'k': 5.120812726526037,
                            'STEPS_pareto': 0.3463158890907987,
                            'N_people_per_space': 40,
                            'RWP_WT_MAX': 317},
                            'clique_with_random': {'p_space_change': 0.026573954773857322,
                            'mean': 271,
                            'sigma': 28,
                            'p_add': 0.038092639210002804,
                            'pareto_shape': 1.592905968448615,
                            'N_people_per_space': 2},
                            'RWP': {'RWP_WT_MAX': 1797},
                            'random': {'p_add': 0.00045426551451094066, 'pareto_shape': 3.08156770075869},
                            'baseline': {'weight': 0.0005436035657689404},
                            'TLW': {'TLW_WT_MAX': 3277,
                            'TLW_WT_EXP': -0.28634300032673937,
                            'FL_MAX': 49,
                            'FL_EXP': -9.894483858860687}}
    
    best_paras_primaryschool = {'STEPS_pareto': {'k': 9.974003647535408,
                                'STEPS_pareto': 0.6130675933667971,
                                'N_people_per_space': 39},
                                'STEPS_with_RWP_pareto': {'k': 7.870151829133263,
                                'STEPS_pareto': 0.5202840800155737,
                                'N_people_per_space': 22,
                                'RWP_WT_MAX': 3600},
                                'clique_with_random': {'p_space_change': 0.06605636980052534,
                                'mean': 514,
                                'sigma': 33,
                                'p_add': 0.046199106031901936,
                                'pareto_shape': 7.2632611210982345,
                                'N_people_per_space': 9},
                                'random': {'p_add': 0.0009190357020863638, 'pareto_shape': 2.455624393138444},
                                'RWP': {'RWP_WT_MAX': 2108},
                                'baseline': {'weight': 0.0013047375595401833},
                                'TLW': {'TLW_WT_MAX': 2307,
                                'TLW_WT_EXP': -0.6106044193554402,
                                'FL_MAX': 85,
                                'FL_EXP': -8.996563363396556}}
    
    best_paras_highschool = {'STEPS_pareto': {'k': 4.3879605793130265,
                                'STEPS_pareto': 0.4210183226476539,
                                'N_people_per_space': 27},
                                'clique_with_random': {'p_space_change': 0.0562632459155923,
                                'mean': 170,
                                'sigma': 91,
                                'p_add': 0.008016298535436495,
                                'pareto_shape': 9.998054452780455,
                                'N_people_per_space': 19},
                                'random': {'p_add': 0.00034618042207039445,
                                'pareto_shape': 7.061619041378239},
                                'baseline': {'weight': 0.0002992849250132354},
                                'STEPS_with_RWP_pareto': {'k': 8.128418593329403,
                                'STEPS_pareto': 0.11968494818705015,
                                'N_people_per_space': 21,
                                'RWP_WT_MAX': 3588},
                                'RWP': {'RWP_WT_MAX': 3599},
                                'TLW': {'TLW_WT_MAX': 2359,
                                'TLW_WT_EXP': -0.5029329300743824,
                                'FL_MAX': 80,
                                'FL_EXP': -9.987833730779137}}

    best_paras_supermarked = {'clique_with_random': {'p_space_change': 0.00820944885032916,
                                'mean': 574,
                                'sigma': 43,
                                'p_add': 0.09139312807734347,
                                'pareto_shape': 2.1211790041196585,
                                'N_people_per_space': 31},
                                'random': {'p_add': 0.009703836159227257, 'pareto_shape': 1.4881299058280244},
                                'STEPS_with_RWP_pareto': {'k': 1.1421931631671745,
                                'STEPS_pareto': 2.0289036323671814,
                                'N_people_per_space': 33,
                                'RWP_WT_MAX': 388},
                                'STEPS_pareto': {'k': 1.1345087010817283,
                                'STEPS_pareto': 2.4205491317513514,
                                'N_people_per_space': 12},
                                'TLW': {'TLW_WT_MAX': 3258,
                                'TLW_WT_EXP': -5.025126349509932,
                                'FL_MAX': 91,
                                'FL_EXP': -0.11181436854902468},
                                'RWP': {'RWP_WT_MAX': 11},
                                'baseline': {'weight': 0.009970022805909147}}
    
    sot = 2000
    new_time = None

    if ntype == 'office':
        capacity = 217
        path = './data_eval_split/InVS/f1_1970-01-01.parquet' 
        max_dist, min_dist, fov = 1.5, 0.0, 2.094395
        SIR_paras = .013, 20 / (7*24*60*60), 2170, 35  # beta, gamma, nruns, ndays
        SIR_path = './results/office_empirical_beta=0.013_gamma=3.306878306878307e-05.npy'
        best_paras = best_paras_office

    elif ntype == 'highschool':
        capacity = 327
        path = './data_eval_split/highschool/f0_2013-12-03.parquet'
        max_dist, min_dist, fov = 1.5, 0.0, 2.094395
        SIR_paras = .007, 20 / (7*24*60*60), 3270, 35
        SIR_path = './results/highschool_empirical_beta=0.007_gamma=3.306878306878307e-05.npy'
        best_paras = best_paras_highschool

    elif ntype == 'primaryschool':
        capacity = 242
        path = './data_eval_split/primaryschool/1970-01-01.parquet'
        max_dist, min_dist, fov = 1.5, 0.0, 2.094395
        SIR_paras = .0013, 20 / (7*24*60*60), 2420, 35
        SIR_path = './results/primaryschool_empirical_beta=0.0013_gamma=3.306878306878307e-05.npy'
        best_paras = best_paras_primaryschool

    elif ntype == 'supermarked':
            capacity = 112.2
            path = './data_eval_split/supermarked/f0_2021-03-17.parquet'
            max_dist, min_dist, fov = 1.5, 0.0, 2.094395
            new_time = 20
            SIR_paras = .075, 20 / (7*24*60*60), 5390, 35
            SIR_path = './results/supermarked_empirical_beta=0.075_gamma=3.306878306878307e-05.npy'
            best_paras = best_paras_supermarked
    
    model_kwargs = best_paras[method]

    if method == 'baseline':
            weight = model_kwargs['weight']
    else:
        weight=None

    try:
        Npps = model_kwargs['N_people_per_space']
        del model_kwargs['N_people_per_space']
    except:
        Npps = 20
    

    EN = EvaluationNetwork(ntype, TU=20, temporal_offset=1000, path=path, switch_off_time=sot, new_time=new_time)
    EN.fov = fov
    EN.contact_dist_min = min_dist
    EN.contact_dist_max = max_dist
    EN.to_tacoma_tn()

    print(f'Location: {ntype}, Method: {method}, Npps: {Npps}, kwargs: {model_kwargs}')

    Loc = cn.build_location(EN.df_trajectories.copy(), loc_id=None, loc_type=ntype, N_pps=Npps, school_space=None, capacity=capacity)
    #EN.cn_approximation(Loc, method, model_kwargs=model_kwargs, export=False,)
    EN.Loc = Loc
    EN.method = method
    for file in os.listdir(f'./networks/{ntype}/medians/'):
        if (f'{method}_median' in file) and (file[0] == method[0]):
            print(file, method)
            tn_approx = tc.load_json_taco(f'./networks/{ntype}/medians/' + file)
            EN.tn_approx = tn_approx
            print(tn_approx.N)


    SIR_save_path = f'./networks/{ntype}/medians_many/{method}_SIR.npy'
    cost = EN.cost_function(approx=True, smoothing=3, weight=weight, SIR_paras=SIR_paras, SIR_path=SIR_path, SIR_save_path=SIR_save_path)
    tc.write_json_taco(tn_approx, f'./networks/{ntype}/medians_many/{method}_SIR_median={cost}.tc')
    print(cost)


class StudyWrapper():
    def __init__(self) -> None:
        self.study = optuna.create_study()
        self.stds = []


if __name__ == '__main__':
    # ntype, path = 'highschool', './data_eval_split/highschool/f0_2013-12-03.parquet'
    # ntype, path = 'office', './data_eval_split/InVS/f1_1970-01-01.parquet' 
    # study = optuna.create_study()
    
    '''space = 120
    ntype = 'highschool' 
    method = 'STEPS_with_RWP_pareto'
    with open(f'{ntype}_tuned_paras_{method}_loc_id=120.pkl', 'rb') as file:
        SW = pickle.load(file)

    #SW = StudyWrapper()

    
    for i in range(50):
        hyperparameter_tuning(space, SW, ntype, method)
        pass'''



    # path = './data_eval_split/gallery/f57_2009-07-04.parquet'
    '''EN = EvaluationNetwork('supermarked')
    EN.to_tacoma_tn()
    # EN.overview_plots()
    EN.eval_df_to_trajectory(180)
    Loc = hm.Location(f'{EN.name}_{EN.name_identifier}', 3, 3, 10, 10)
    EN.hm_approximation(Loc, 'RWP', 20)
    EN.overview_plots(True)'''

    #collect_supermarked()
    # print(df)


    '''# Plot empirical results
    time = np.arange(4321*50) * 20/(24*60*60)
    I_emp = np.load('./results/1_empirical_beta=0.007_gamma=3.306878306878307e-05.npy')
    I_emp = I_emp[:, :-101]
    I_mean_emp, I_err_upper_emp, I_err_lower_emp = util.mean_with_errors(I_emp, 330)

    plt.plot(time, I_mean_emp / 330, label='data')
    plt.legend()

    plt.fill_between(time, I_err_lower_emp / 330, I_err_upper_emp / 330, alpha=.3)
    #plt.savefig(f'./tests/highschool_SIR_comparison_STEPS_Npps={Npps}.png')'''


    #parameter_sweeps_STEPS(k=1.5, Npps=20)

    #ntype, path = 'office', './data_eval_split/InVS/f0_1970-01-01.parquet'

    
    
    '''
    school_space = 10

    Npps, p_space_change, mean, sigma, p_add, pareto_shape = 20, 1/100, 3600, 5, .003, 2.5
    model_kwargs = {'N_people_per_space': Npps, 'p_space_change': p_space_change, 'mean': mean, 'sigma': sigma, 'p_add': p_add, 'pareto_shape': pareto_shape}
        
    Loc_hs = cn.build_location(EN.df_trajectories.copy(), loc_id=school_space, loc_type='school', N_pps=Npps, school_space=school_space)
    # print(model_kwargs)
    EN.cn_approximation(Loc_hs, 'clique_with_random', model_kwargs=model_kwargs, export=False)
    print(len(EN.tn.t), len(EN.tn_approx.t))
    cost = EN.overview_plots(approx=True, smoothing=3)'''
    #print(cost)
    
    '''
    #ntype, path = 'highschool', './data_eval_split/highschool/f0_2013-12-03.parquet'
    ntype, path, capacity = 'office', './data_eval_split/InVS/f1_1970-01-01.parquet', 217
    #ntype, path, best_paras = 'primaryschool', './data_eval_split/primaryschool/1970-01-01.parquet', None
    # ntype, path = 'supermarked', './data_eval_split/supermarked/f0_2021-03-17.parquet'
    #school_space, best_paras = 10, best_paras_10
    space = None

    if ntype == 'supermarked':
        new_time = 20
    else:
        new_time = None

    #for method in ['STEPS_with_RWP_pareto', 'STEPS_pareto', 'random', 'clique', 'clique_with_random', 'RWP', 'TLW', 'baseline']:
    for method in ['STEPS_with_RWP_pareto']:
        EN = EvaluationNetwork(ntype, TU=20, temporal_offset=1000, path=path, switch_off_time=2000)
        EN.to_tacoma_tn()
        EN.fov = 2.094395
        EN.contact_dist_min = 0.0
        EN.contact_dist_max = 1.5

        # model_kwargs = best_paras[method]
        Npps = 39
        model_kwargs = {'k': 8.195624092106815, 'STEPS_pareto': 0.25453824669772673, 'RWP_WT_MAX': 263}

        if method == 'baseline':
            weight = model_kwargs['weight']
        else:
            weight=None

        try:
            Npps = model_kwargs['N_people_per_space']
            del model_kwargs['N_people_per_space']
        except:
            Npps = 20
            

        Loc = cn.build_location(EN.df_trajectories.copy(), capacity=217, loc_id=space, loc_type=ntype, N_pps=Npps, school_space=space)
        EN.cn_approximation(Loc, method, model_kwargs=model_kwargs, export=False)

        cost = EN.overview_plots(approx=True, smoothing=3, weight=weight, ntype=ntype)
    print(cost)'''
    #print(EN.cost_function)

    
    #EN.overview_mult(method, model_kwargs, school_space, Npps=Npps, nruns=101, networks=networks)
    
    for method in ['STEPS_pareto']:    
        #best_para_many_runs('office', method)
        generate_best_para_realizations('highschool', method)
        

    
        
        
    


    pass

    


