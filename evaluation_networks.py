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
from scipy.stats import ks_2samp
import pyreadr
import teneto


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
    def __init__(self, name, path=None, new_time=None):
        self.name = name

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
            print(self.df[['i', 'j', 't']])
        else:
            self.new_time = 1

        # Setting minimum distance for contact to be counted
        self.contact_dist = 2.0


    def to_tacoma_tn(self):
        tn = tc.edge_lists()
        tn.N = max(self.df.i.max(), self.df.j.max()) + 1
        Nt = self.df.t.max() + 1
        tn.t = list(range(Nt))
        tn.tmax = Nt
        tn.time_unit = '20s'

        contacts = [[] for _ in range(Nt)]

        for _, contact in self.df.iterrows():
            contacts[contact.t].append([contact.i, contact.j])
        
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

                # Generate trajectories
                for _, (sonp, sofp) in enumerate(zip(switch_off_points[:-1], switch_off_points[1:])):
                    p_id.append(person_contact.ij)
                    activity_start_min.append(person_contact.tt[sonp + 1])
                    activity_end_min.append(person_contact.tt[sofp])
                
            self.df_trajectories = pd.DataFrame({'p_id': p_id, 'activity_start_min': activity_start_min, 'activity_end_min': activity_end_min})
            return self.df_trajectories
    

    def cn_approximation(self, Loc, method, model_kwargs=None):
        self.method = method
        if method == 'random':
            self.new_time = 1
        ts, te = self.df_trajectories.activity_start_min.min(), self.df_trajectories.activity_end_min.max() 
        CN = cn.ContactNetwork(self.df_trajectories, Loc, ts, te, self.new_time)

        # Set parameters of model
        if model_kwargs:
            for par, value in model_kwargs.items():
                setattr(CN, par, value)

        CN.make_movement(method=method)
        print('finished simulation \nstart working on network')

        self.tn_approx = CN.make_tacoma_network(self.contact_dist, self.new_time)
    
    def to_teneto(self):
        pass
            

    def overview_plots(self, approx, ind=None, smoothing=60):
        if approx:
            networks = [self.tn, self.tn_approx]
        else:
            networks = [self.tn]
        
        labels = [['contact', 'inter contact'], ['model contact', 'model inter contact']]
        colors = ['#1f77b4', '#ff7f0e']
        labels2 = ['data', 'model']
        ICTs, CDs, degrees = [], [], []

        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.name} {self.name_identifier}')
        (ax1, ax2, ax3, ax4) = axs.flatten()
        for i, tn in enumerate(networks):
            res = tc.api.measure_group_sizes_and_durations(tn)
            ICTs.append(res.group_durations[1])
            CDs.append(res.contact_durations)
            plot_contact_durations(res, (ax1, ax3), fit_power_law=True, bins=100, xlabel='duration [min]', color=colors[i], label=labels[i])

            ax2.set_title('time average degreee distribution')
            degree = np.array(tc.api.degree_distribution(tn))
            degree = degree[degree > 0]
            degrees.append(degree)
            plot_degree_distribution(degree, ax2, label=labels2[i])

            _, _, m = tc.edge_counts(tn)
            m, t = moving_average(m[:-1], smoothing), moving_average(tn.t, smoothing)
            ax4.set_title('rolling average edge_counts')
            ax4.plot(t, m, color=colors[i], label=labels2[i], alpha=.5)

        ks_test_ICTs = ks_2samp(ICTs[0], ICTs[1], alternative='tow-sided')
        ks_test_CDs = ks_2samp(CDs[0], CDs[1], alternative='two-sided')
        ks_test_degrees = ks_2samp(degrees[0], degrees[1])
        cd_title = 'contact duration distribution, KS: {statistic: .2f} p={pval: .2g}'
        ict_title = 'inter contact time distribution, KS: {statistic: .2f} p={pval: .2g}'
        degree_title = r'$\bar d(t)$, ' + 'KS: {statistic: .2f} p={pval: .2g}'

        ax1.set_title(cd_title.format(statistic=ks_test_CDs.statistic, pval=ks_test_CDs.pvalue))
        ax3.set_title(ict_title.format(statistic=ks_test_ICTs.statistic, pval=ks_test_ICTs.pvalue))
        ax2.set_title(degree_title.format(statistic=ks_test_degrees.statistic, pval=ks_test_degrees.pvalue))

        for ax in axs.flatten():
            ax.legend()

        plt.tight_layout()
        plt.savefig(f'./plots/eval_networks/overview_{self.name}_{self.name_identifier}_approx_{approx}_{self.method}_{ind}.png')
        plt.close()

        return ks_test_CDs.statistic, ks_test_CDs.pvalue, ks_test_ICTs.statistic, ks_test_ICTs.pvalue


if __name__ == '__main__':
    # path = './data_eval_split/gallery/f57_2009-07-04.parquet'
    '''EN = EvaluationNetwork('supermarked')
    EN.to_tacoma_tn()
    # EN.overview_plots()
    EN.eval_df_to_trajectory(180)
    Loc = hm.Location(f'{EN.name}_{EN.name_identifier}', 3, 3, 10, 10)
    EN.hm_approximation(Loc, 'RWP', 20)
    EN.overview_plots(True)'''
    #collect_supermarked()
    path = './data_eval_split/supermarked/'
    files = os.listdir(path)

    EN = EvaluationNetwork('supermarked', path + files[0], 20)
    EN.to_tacoma_tn()
    EN.eval_df_to_trajectory(None)

    n_space = 10
    rwp_wt_max = 5
    v_RWP_min = .1
    v_RWP_max = 1.
    Loc = cn.Location(0, n_space, n_space, 3.1, 3.1)
    model_kwargs = {'RWP_WT_MAX': rwp_wt_max, 'v_RWP_min': v_RWP_min, 'v_RWP_max': v_RWP_max}
    EN.cn_approximation(Loc, 'RWP', model_kwargs)
    res = EN.overview_plots(True, 4)

    pass

    


