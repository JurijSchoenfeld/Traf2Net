# Used for plotting networks

import notebook_archive.create_network as cn
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


def plot_all_graphs():
    '''In this function an overview plot, for all networks created in the create network module, is made.'''

    all_Graphs, names = cn.create_network14_dataset()
    n_axs = int(np.ceil(np.sqrt(len(names))))

    print(n_axs)

    fig, axs = plt.subplots(n_axs, n_axs, figsize=(20, 20))
    axs = axs.flatten()

    for G, ax, name in zip(all_Graphs, axs, names):
        print(name)
        nx.draw_networkx(G, ax=ax)
        ax.set_title(name)
    
    plt.savefig('./plots/graph_overview.png')



def plot_singular_graph(other_metrics=True):
    '''Aim of this function is to provide a detailed overview of topoligal characteristics of all networks created in the create network module.
       Certain metrics may be added or removed in the future. '''
    
    all_Graphs, names = cn.create_network14_dataset(exclude=['Enron', 'Facebook', 'Sex', 'PB', 'sfhh', 'WV'])

    n_axs = 1
    if other_metrics:
        n_axs = 4

    for G, name in zip(all_Graphs, names):
        print(name)
        fig, axs = plt.subplots(3, 4, figsize=(20, 20))
        axs = np.array([[axs]]).flatten()

        nx.draw_networkx(G, ax=axs[0], node_size=50, alpha=.2, with_labels=False)
        axs[0].set_title(name, fontsize=20)

        if other_metrics:
            y_degree_hist = nx.degree_histogram(G)
            axs[1].bar(np.arange(0, len(y_degree_hist)), y_degree_hist)
            axs[1].set_title('degree distribution', fontsize=20)

            y_average_degree_connectivity = nx.average_degree_connectivity(G)
            axs[2].scatter(x= list(y_average_degree_connectivity.keys()), y=list(y_average_degree_connectivity.values()))
            axs[2].set_title('average degree connectivity', fontsize=20)

            axs[3].hist(list(nx.degree_centrality(G).values()))
            axs[3].set_title('degree centrality', fontsize=20)

            axs[4].hist(list(nx.eigenvector_centrality(G).values()))
            axs[4].set_title('eigenvector centrality', fontsize=20)

            axs[5].hist(list(nx.betweenness_centrality(G).values()))
            axs[5].set_title('betweenness centrality', fontsize=20)

            axs[6].hist(list(nx.closeness_centrality(G).values()))
            axs[6].set_title('closeness centrality', fontsize=20)

            axs[7].hist(list(nx.current_flow_closeness_centrality(G).values()))
            axs[7].set_title('current flow closeness centrality', fontsize=20)

            axs[8].hist(list(nx.current_flow_betweenness_centrality(G).values()))
            axs[8].set_title('current flow betweenness centrality', fontsize=20)

            axs[9].hist(list(nx.katz_centrality_numpy(G).values()))
            axs[9].set_title('katz centrality', fontsize=20)

            axs[10].hist(list(nx.clustering(G).values()))
            axs[10].set_title('clustering', fontsize=20)




        plt.savefig(f'./plots/{name}.png')
    

def clustering_against_nn_degree():
    '''Investigate wether there exits a correlation between clustering coefficient of a certain node and it's
       average nearest neighbour degree. If this is true, it could be possible to estimate the risk of infection 
       from a nodes clustering coefficient as a high average nn degree is expected to increase infection risk.'''
    
    all_Graphs, names = cn.create_network14_dataset(exclude=['Sex'])
    del all_Graphs[1:3]
    del names[1:3]

    fig, axs = plt.subplots(4, 4, figsize=(20, 15))
    axs = axs.flatten()
    for G, name, ax in zip(all_Graphs, names, axs):
        print(name)
        nn_degree = np.array(list(nx.average_neighbor_degree(G).values()))
        clustering_coef = np.array(list(nx.clustering(G).values()))
        degrees = np.array(nx.degree(G))[:, 1]
        max_degree = degrees.max()
        rel_degree = degrees / max_degree
        threshold = .05
        mask = rel_degree >= threshold
        cor_coef, p_val = pearsonr(clustering_coef[mask], nn_degree[mask])

        ax.set_title(f'{name}, r={round(cor_coef, 2)}, p={round(p_val, 3)}')
        ax.scatter(clustering_coef[mask], nn_degree[mask], alpha=.2)
        ax.set_xlabel(r'$c_i$')
        ax.set_ylabel(r'$k_{nn, i}$')
    
    plt.tight_layout()
    plt.savefig(f'./plots/nndegree_clustering/nndegree_clustering.png')


def clustering_against_corenumber():
    all_Graphs, names = cn.create_network14_dataset(exclude=['Sex'])
    del all_Graphs[1:3]
    del names[1:3]

    fig, axs = plt.subplots(4, 4, figsize=(20, 15))
    axs = axs.flatten()
    for G, name, ax in zip(all_Graphs, names, axs):
        print(name)
        core_number = np.array(list(nx.core_number(G).values()))
        clustering_coef = np.array(list(nx.clustering(G).values()))
        degrees = np.array(nx.degree(G))[:, 1]
        max_degree = degrees.max()
        rel_degree = degrees / max_degree
        threshold = .05
        mask = rel_degree >= threshold
        cor_coef, p_val = pearsonr(clustering_coef[mask], core_number[mask])

        ax.set_title(f'{name}, r={round(cor_coef, 2)}, p={round(p_val, 3)}')
        ax.scatter(clustering_coef, core_number, alpha=.2)
        ax.set_xlabel(r'$c_i$')
        ax.set_ylabel(r'core number')
    
    plt.tight_layout()
    plt.savefig(f'./plots/corenumber_clustering/corenumber_cluster.png')


def clustering_against_nndegree_1p5setting():
    all_Graphs, names = cn.create_network14_dataset(exclude=['Sex'])
    del all_Graphs[1:3]
    del names[1:3]

    fig, axs = plt.subplots(4, 4, figsize=(20, 15))
    axs = axs.flatten()





if __name__ == '__main__':
    # plot_singular_graph()
    # clustering_against_nn_degree()
    pass
