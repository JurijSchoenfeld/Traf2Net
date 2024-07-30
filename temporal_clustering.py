import tacoma as tc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import evaluation_networks as en
import contact_networks as cn
import os
import util
from math import sqrt, ceil
from util import plot_contact_durations
from tacoma.analysis import plot_degree_distribution
import dynetx as dn
import cdlib
from cdlib import algorithms, viz, evaluation, TemporalClustering
import networkx as nx
import tacoma as tc
import warnings
from scipy.stats.stats import pearsonr


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
            EN.to_tacoma_tn()
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
    
    def approximate_network(self, switch_off_time, method, Loc, model_kwargs, overview=False):
        self.ApproxTC = TemporalClustering()

        # Approximate observed network with given method and parameters
        nx_graphs_all_days = []
        for EN in self.EN_list:
            EN.eval_df_to_trajectory(switch_off_time)
            EN.cn_approximation(Loc, method, model_kwargs)
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


if __name__ == '__main__':
    STC_RANDOM = SuperTemporalClustering([en.EvaluationNetwork('supermarked', path + file, 20) for file in files[:6]])
    STC_RANDOM.detect_communities(algorithms.louvain, kwargs={'resolution': 1.})

    n_space = 10
    p_add, pareto_shape = .02, 2.0

    Loc = cn.Location(0, n_space, n_space, 3.1, 3.1)
    model_kwargs = {'p_add': p_add, 'pareto_shape': pareto_shape}

    STC_RANDOM.approximate_network(None, 'random', Loc, model_kwargs, overview=False)
    STC_RANDOM.detect_communities(approx=True)
    