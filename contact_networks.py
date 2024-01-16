import numpy as np
import pandas as pd
import util
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.markers import MarkerStyle
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from scipy.spatial import KDTree
from scipy.sparse import triu, coo_array
from scipy.spatial.distance import pdist, squareform
import random
from math import ceil, floor
from mobility import truncated_levy_walk, random_waypoint
import tacoma as tc
import clique


def attractor_pdf(x, k):
    return (k - 1)/(1+x)**k


def attractor_icdf(k):
    # Returns a sample drawn from the attractor pdf
    p = random.uniform(0, 1)
    return (1-p)**(1/(1-k)) - 1


class Node:
    def __init__(self, node_id, tmax):
        self.id = node_id

        self.x = np.empty(tmax)
        self.x.fill(np.nan)

        self.y = np.empty(tmax)
        self.y.fill(np.nan)

        self.space = np.empty(tmax)
        self.space.fill(np.nan)

        # self.time = np.arange(tmin, tmax, 1)


class Space:
    def __init__(self, space_id, xlen, ylen):
        self.space_id = space_id
        i, j = space_id

        self.centroids_x = (j + .5) * xlen
        self.centroids_y = (i + .5) * ylen
        self.space_bounds_x = (self.centroids_x - .5 *xlen, self.centroids_x + .5 *xlen)
        self.space_bounds_y = (self.centroids_y - .5 *ylen, self.centroids_y + .5 *ylen)
        self.nodes = []
    
    def add_node(self, Node):
        self.nodes.append(Node)

    def remove_node(self, Node):
        self.nodes.remove(Node)
    
    def get_random_coords(self):
        # Return a random point in space
        return random.uniform(*self.space_bounds_x), random.uniform(*self.space_bounds_y)


class Location:
    def __init__(self, loc_id, spaces_x, spaces_y, space_dim_x, space_dim_y):
        self.N_zones = spaces_x * spaces_y
        self.space_dim_x = space_dim_x
        self.space_dim_y = space_dim_y
        self.loc_id = loc_id
        self.spaces_x = spaces_x
        self.spaces_y = spaces_y

        self.spaces = []
        self.space_dict = {}

        for i in range(spaces_y):
            for j in range(spaces_x):
                self.spaces.append(Space((i, j), self.space_dim_x, self.space_dim_y))
                self.space_dict[(i, j)] = len(self.spaces)

    def plot_location(self, fig, ax, centroids=False):
        # Base plot of location with 
        for space in self.spaces:
            ax.hlines([*space.space_bounds_y], space.space_bounds_x[0], space.space_bounds_x[1], color='grey', alpha=.3)
            ax.vlines([*space.space_bounds_x], space.space_bounds_y[0], space.space_bounds_y[1], color='grey', alpha=.3)
            
            if centroids:
                ax.scatter([space.centroids_x], [space.centroids_y], color='tab:blue', marker=MarkerStyle(marker='s', fillstyle='none'))
    
    def get_distance_matrix(self):
        # Return distances between spaces centroids in distance matrix form
        X = [(space.centroids_x, space.centroids_y) for space in self.spaces]

        self.distance_matrix = squareform(pdist(X))


class ContactNetwork:
    def __init__(self, df, Location, t_start, t_end, time_scale_data=60):
        '''
        df: a pandas data frame where every row is trajectory concluding in Location, 
        with the following coloumns:
            p_id: a unique identifier mapping to a individual
            activity_start_min/activity_end_min: start/end time of the activity in the Location
            activity_name_mct: describtion of the activity done at location loc_id_end
        
        Location: An instance of Location class defining important parameters for simulation

        t_start, t_end: start/end time of the simulation. 

        time_scale_data: The number of seconds one time unit in the df represents
        '''
        self.Location = Location
        self.df = df
        # FOR TESTING!!! Simulate only a few nodes
        # self.df = self.df.head(100)

        # map p_id to index like
        util.map_pid(self.df)

        # Setting parameters
        self.time_scale_data = time_scale_data
        self.unique_nodes = self.df.p_id.unique()
        self.t_start = int(t_start * self.time_scale_data) # convert to seconds
        self.t_end = ceil(t_end * self.time_scale_data) # convert to seconds
        self.df.activity_start_min = (self.df.activity_start_min * self.time_scale_data).round().astype('int') # convert to seconds
        self.df.activity_end_min = (self.df.activity_end_min * self.time_scale_data).round().astype('int') # convert to seconds

        # Initialize nodes this will be done on simulation level later
        self.nodes = [Node(node, self.t_end) for node in np.arange(0, self.df.p_id.max() + 1)]

        # Model parameters
        self.k = 1.2 # Strenght of the attractor must be greater than 1
        self.STEPS_pause_min, self.STEPS_pause_max = 20, 30
        self.v_STEPS_min, self.v_STEPS_max = .83, 3.2 # interval for uniform distribution to pick travel speed between zones from
        self.v_RWP_min, self.v_RWP_max = .1, 1. # interval for uniform distribution to pick travel speed between waypoints from
        self.RWP_tpause_min, self.RWP_tpause_max = 0, 5 # ranges to pick waiting time in waypoint from
        self.tlw_max_wt = 100
        self.min_contact_duration = None
        self.p_add, self.pareto_shape = .001, 3.
        self.N_peaoplePerSpace, self.p_space_change, self.mean, self.sigma = 15, 1/100, 10, 5

        print('Initialized contact network model')


    def RWP_main(self, nagents, dim):
        # Returns RWP positions for n-agents moving in a room with dimensions dim
        rwp = random_waypoint(nagents, dimensions=dim, wt_max=5)
        pos = np.array([next(rwp).copy() for _ in range(self.t_end - self.t_start)])
        return pos


    def STEPS_with_RWP(self, row, pos):
        TS, TE = row.activity_start_min, row.activity_end_min
        Z0 = row.default_zone
        NODE = self.nodes[row.p_id]
        agent_pos = pos[:, row.p_id, :]
        
        # Initialize
        t = TS
        z = Z0
        xp, yp = self.Location.spaces[Z0].get_random_coords()

        while t < TE:
            # Choose new space
            possible_new_z, = np.where(self.Location.distance_matrix[Z0] <= attractor_icdf(self.k))
            z = np.random.choice(possible_new_z, 1)[0]

            # Prepare RWP
            tpause = random.randint(self.STEPS_pause_min, self.STEPS_pause_max)  # this bit is not in line with the STEPS paper, instead it uses the probability distribution from their Matlab code 
            if TE - TS - tpause <= 0:
                tpause = TE - TS
                
            a = random.randint(0, TE - TS - tpause)
            rwp_pos = agent_pos[a: a + tpause]
            rwp_pos_x = rwp_pos[:, 0] + self.Location.spaces[z].space_bounds_x[0]
            rwp_pos_y = rwp_pos[:, 1] + self.Location.spaces[z].space_bounds_y[0]

            # Travel to new space
            # Calculate travel trajectory
            x, y = rwp_pos_x[0], rwp_pos_y[0]
            distance = ((x - xp)**2 + (y - yp)**2)**.5
            travel_speed = random.uniform(self.v_STEPS_min, self.v_STEPS_max)
            travel_time = round(distance / travel_speed)

            if travel_time <= 1:
                travel_time = 1
                x_interpolated = np.array([x])
                y_interpolated = np.array([y])
            else:
                x_interpolated = np.linspace(xp, x, travel_time)
                y_interpolated = np.linspace(yp, y, travel_time)

            if travel_time + t > TE:
                # TODO: Find a smarter way to end trajectory in the future
                break

            NODE.x[t: t + travel_time] = x_interpolated
            NODE.y[t: t + travel_time] = y_interpolated

            if t + travel_time + tpause > TE:
                # TODO: Find a smarter way to end trajectory in the future
                break

            # RWP
            NODE.x[t + travel_time: t + travel_time + tpause] = rwp_pos_x
            NODE.y[t + travel_time: t + travel_time + tpause] = rwp_pos_y
            
            # Update
            t = t + travel_time + tpause
            xp, yp = rwp_pos_x[-1], rwp_pos_y[-1]


    def STEPS(self, row):
        TS, TE = row.activity_start_min, row.activity_end_min
        Z0 = row.default_zone
        NODE = self.nodes[row.p_id]
        
        # Initialize
        t = TS
        tpause = random.randint(self.STEPS_pause_min, self.STEPS_pause_max)  # this bit is not in line with the STEPS paper, instead it uses the probability distribution from their Matlab code 

        if t + tpause > TE:
            # TODO: Find a smarter way to end trajectory in the future
            return

        NODE.space[t: t + tpause] = Z0
        x0, y0 = self.Location.spaces[Z0].get_random_coords()
        NODE.x[t: t + tpause] = x0
        NODE.y[t: t + tpause] = y0

        # Update iterators
        xp, yp = x0, y0
        t = t + tpause

        while t < TE:
            # Choose new Space
            possible_new_z, = np.where(self.Location.distance_matrix[Z0] <= attractor_icdf(self.k))
            z = np.random.choice(possible_new_z, 1)[0]
            x, y = self.Location.spaces[z].get_random_coords()
            tpause = random.randint(self.STEPS_pause_min, self.STEPS_pause_max)  # this bit is not in line with the STEPS paper, instead it uses the probability distribution from their Matlab code 

            # Calculate travel trajectory
            distance = ((x - xp)**2 + (y - yp)**2)**.5
            travel_speed = random.uniform(self.v_STEPS_min, self.v_STEPS_max)

            travel_time = round(distance / travel_speed)

            if travel_time + t > TE:
                # TODO: Find a smarter way to end trajectory in the future
                break

            if travel_time == 0:
                NODE.x[t] = x
                NODE.y[t] = y
            else:
                x_interpolated = np.linspace(xp, x, travel_time)
                y_interpolated = np.linspace(yp, y, travel_time)
            
                NODE.x[t: t + travel_time] = x_interpolated
                NODE.y[t: t + travel_time] = y_interpolated
        
            if travel_time + tpause + t > TE:
                tpause = TE - (t + travel_time)

            # Set new Space
            NODE.space[t + travel_time: t + travel_time + tpause] = z

            # no RWP
            NODE.x[t + travel_time: t + travel_time + tpause] = x
            NODE.y[t + travel_time: t + travel_time + tpause] = y
            xp, yp = x, y

            # Update time
            t = t + travel_time + tpause


    def TLW(self, nagents):
        dim =(self.Location.space_dim_x * self.Location.spaces_x, self.Location.space_dim_y * self.Location.spaces_x)
        tlw = truncated_levy_walk(nagents, dimensions=dim, WT_MAX=self.tlw_max_wt)
        pos = np.array([next(tlw).copy() for _ in range(self.t_end - self.t_start)])
        return pos


    def baseline(self):
        start_times, end_times = self.df.activity_start_min.values, self.df.activity_end_min.values
        event_ids = self.df.p_id.values
        loc_id_end = self.df.loc_id_end.values[0]
        # Takes a df containing all activities at location
        # Returns all possible contacts

        # Broadcast the start_time and end_time arrays for comparison with each other
        overlap_start = np.maximum.outer(start_times, start_times)
        overlap_end = np.minimum.outer(end_times, end_times)

        # Calculate the overlap duration matrix 
        overlap_durations = np.maximum(overlap_end - overlap_start, np.zeros(shape=overlap_start.shape)).astype('uint16')
        
        # Set lower triangle and main diagonal to zero (overlap of an event with itself and double counting)
        overlap_durations = np.triu(overlap_durations, 1)

        # Extract contact rows, cols
        rows, cols = np.where(overlap_durations > 0)
        p_A = event_ids[rows].astype('int')

        # Save contacts to new DataFrame
        df_contacts = pd.DataFrame({'p_A': p_A,'p_B': event_ids[cols].astype('int'), 
                        'start_of_contact': overlap_start[rows, cols].astype('int'),
                        'end_of_contact': overlap_end[rows, cols].astype('int'),
                        'loc_id': np.repeat(loc_id_end, len(p_A)).astype('int32')})
        
        # Calculate contact durations
        df_contacts['contact_duration'] = df_contacts.end_of_contact - df_contacts.start_of_contact

        # (optional) drop contacts below specified contact duration
        if self.min_contact_duration:
            df_contacts = df_contacts[df_contacts.contact_duration >= self.min_contact_duration]
        
        return df_contacts


    def random(self, contact):
        # TODO: implement
        tstart, tend = contact.start_of_contact, contact.end_of_contact

        # Select contacts randomly
        selected_contact = np.full(tend - tstart, False) 
        # Random start times
        start = np.where(np.random.rand(tend - tstart) <= self.p_add)[0]
        # Random durations
        duration = np.ceil(np.random.pareto(self.pareto_shape, len(start))).astype('int')
        duration[duration <= 0] = 1  # contact duration must be at least one time step
        end = start + duration

        # Avoid multi edges
        for s, e in zip(start, end):
            selected_contact[s: e + 1] = True
        
        selected_contact_start, selected_contact_end = util.boolean_blocks_indices(selected_contact)
        selected_contact_start += tstart
        selected_contact_end += tstart

        n = len(selected_contact_start)
        return pd.DataFrame({'p_A': np.full(n, contact.p_A), 'p_B': np.full(n, contact.p_B), 'start_of_contact': selected_contact_start, 'end_of_contact': selected_contact_end})


    def get_positions(self, grp, pos):
        # Helper function to get positions from mobility.py generator
        indices = np.concatenate(grp.to_numpy()) - 1
        self.nodes[grp.name].x[indices] = pos[indices, grp.name, 0]
        self.nodes[grp.name].y[indices] = pos[indices, grp.name, 1]


    def make_movement(self, method):
        print('Start making movement')
        self.METHOD = method

        if method in ['baseline', 'random', 'clique']:
            print(f'{method} is not movement based. Return')
            return

        if method == 'STEPS':
            # Assign default zones to all pid
            self.Location.get_distance_matrix()
            default_zones_map = dict(zip(self.unique_nodes, np.random.randint(0, self.Location.N_zones, self.unique_nodes.shape[0])))
            self.df['default_zone'] = self.df.p_id.map(default_zones_map)
            self.df.apply(self.STEPS, axis=1)
        
        elif method == 'STEPS_with_RWP':
            # get RWP positions
            dim = (self.Location.space_dim_x, self.Location.space_dim_y)
            pos = self.RWP_main(len(self.unique_nodes), dim=dim)
        
            # Assign default zones to all pid
            self.Location.get_distance_matrix()
            default_zones_map = dict(zip(self.unique_nodes, np.random.randint(0, self.Location.N_zones, self.unique_nodes.shape[0])))
            self.df['default_zone'] = self.df.p_id.map(default_zones_map)
            # get STEPS positions
            self.df.apply(self.STEPS_with_RWP, pos=pos, axis=1)
        
        elif method == 'TLW':
            pos = self.TLW(len(self.unique_nodes))
            result = self.df.apply(util.generate_array, axis=1)
            result.groupby(self.df['p_id']).apply(self.get_positions, pos=pos)
        
        elif method == 'RWP':
            dim = (self.Location.space_dim_x * self.Location.spaces_x, self.Location.space_dim_y * self.Location.spaces_x)
            pos = self.RWP_main(len(self.unique_nodes), dim)

            result = self.df.apply(util.generate_array, axis=1)
            result.groupby(self.df['p_id']).apply(self.get_positions, pos=pos)


    def network_animation(self, pos):
        # Make network and retun segments, dist for animation
        posTree = KDTree(pos)
        relevant_distances = triu(posTree.sparse_distance_matrix(posTree, max_distance=5, p=2), k=1)

        Aind, Bind, dist = relevant_distances.row, relevant_distances.col, relevant_distances.data
        nodeA = pos[Aind]
        nodeB = pos[Bind]

        segments = np.stack((nodeA, nodeB), axis=1)
        return segments, dist


    def tn_from_contacts(self, contacts):
        # Initilize tacoma temporal network
        tn = tc.edge_changes()
        tn.N = len(self.unique_nodes)
        tmax, tmin = contacts.end_of_contact.max(), contacts.start_of_contact.min()
        Nt = tmax - tmin + 1
        tn.t = list(range(tmin, tmax + 1))
        tn.tmax = tmax + 1
        tn.time_unit = '60s'

        # Make edges
        edges_in, edges_out = [[] for _ in range(Nt)], [[] for _ in range(Nt)]
        for _, c in contacts[['p_A', 'p_B', 'start_of_contact', 'end_of_contact']].iterrows():
            edges_in[c.start_of_contact - tmin].append([c.p_A, c.p_B])
            edges_out[c.end_of_contact - tmin].append([c.p_A, c.p_B])

        tn.edges_in = edges_in
        tn.edges_out = edges_out

        # Check for errors
        print('edge changes errors: ', tc.verify(tn))

        return tn


    def make_tacoma_network(self, max_dist=2, time_resolution=20):
        print('Start network construction')
        if self.METHOD in ['RWP', 'TLW', 'STEPS', 'STEPS_with_RWP']:
            # Get node positions
            X = np.array([node.x for node in self.nodes])
            Y = np.array([node.y for node in self.nodes])
            all_pos = np.array((X, Y)).T

            # For testing, take only positions from a few nodes
            # all_pos = all_pos[:100] 
            
            # Initiate tacoma dynamic network
            tn = tc.edge_lists()
            tn.N = len(self.unique_nodes)
            Nt = ceil((self.t_end - self.t_start)/time_resolution)
            tn.t = list(range(Nt))
            tn.tmax = Nt
            tn.time_unit = '20s'
            contacts = []
            relevant_distances = coo_array((tn.N, tn.N))

            for t, pos in enumerate(all_pos):
                posTree = KDTree(pos)
                # Add all contacts together
                relevant_distances = relevant_distances + triu(posTree.sparse_distance_matrix(posTree, max_distance=max_dist, p=2), k=1)

                if (t + 1) % time_resolution == 0:
                    # Keep only contacts that occured at least once during the time window
                    Aind, Bind = relevant_distances.nonzero()
                    contacts.append(list(zip(Aind, Bind)))
                    relevant_distances = coo_array((tn.N, tn.N))
                
                if t % 10000 == 0:
                    print(f'{t}/{self.t_end}')

            # Check for errors and convert to edge_changes
            tn.edges = contacts
            print('edge list errors: ', tc.verify(tn))

            tn = tc.convert(tn)
            print('edge changes errors: ', tc.verify(tn))

            return tn

        if self.METHOD in ['baseline', 'random', 'clique'] and self.time_scale_data != 60:
            print(f'{self.METHOD} only supports time_scale_data = 60 \nUpdate time resolution of your data and set time_scale_data accordingly or choose a different method')
            return


        elif self.METHOD == 'baseline':
            # Get contacts
            contacts = self.baseline()

            return self.tn_from_contacts(contacts)

        elif self.METHOD == 'random':
            # Get all possible contacts
            contacts = self.baseline()

            # Select random contacts
            selected_contacts = contacts.apply(self.random, axis=1)
            selected_contacts = pd.concat(selected_contacts.to_list(), ignore_index=True)
            
            return self.tn_from_contacts(selected_contacts)

        elif self.METHOD == 'clique':
            # Select contacts if nodes share space
            space_contacts = clique.get_contacts_spaces(self.df, self.N_peaoplePerSpace, self.p_space_change, self.mean, self.sigma)
            return self.tn_from_contacts(space_contacts)


    def animate_movement(self):
        print('Start animation')
        if self.METHOD in ['baseline', 'random', 'clique']:
            print(f'The specified method: {self.METHOD} does not support animation')
            return
        
        fig, ax = plt.subplots(figsize=(9, 9))
        self.Location.plot_location(fig, ax)

        X = np.array([node.x for node in self.nodes])
        Y = np.array([node.y for node in self.nodes])
        pos = np.array((X, Y)).T
        norm = Normalize(vmin=0, vmax=5)
        segments, dist = self.network_animation(pos[0])
        lc = LineCollection(segments, cmap='Reds_r', norm=norm, linewidth=2)
        lc.set_array(dist)
        norm = Normalize(vmin=0, vmax=5)
        ax.add_collection(lc)
        scat = ax.scatter(pos[:, 0], pos[:, 1], c='grey')
        cbar = plt.colorbar(lc, ax=ax)
        cbar.set_label('distance [m]')

        frame_start, frame_stop = round((self.t_end - self.t_start)/2), round((self.t_end - self.t_start)/2 + 300)  # animate 300 frames

        def animate(frame):
            if frame % 10 == 0:
                print(f'{frame}/{frame_stop}')

            # update nodes
            x = X[:, frame]
            y = Y[:, frame]
            data = np.stack([x, y]).T
            scat.set_offsets(data)
            ax.set_title(f'{self.METHOD}, TU: {int(frame)}, 5xTU/s')

            # update edges
            segments, dist = self.network_animation(pos[frame])
            lc.set_segments(segments)
            lc.set_array(dist)

        anim = FuncAnimation(fig, animate, range(frame_start, frame_stop), interval=200)
        anim.save(f'./plots/human_mobility/{self.Location.loc_id}_{self.METHOD}_animation_test.gif')


def interpolation_test_singular(HumanMobilityModel, t0, tend):
    if HumanMobilityModel.METHOD in ['baseline', 'random', 'clique']:
        print(f'{HumanMobilityModel.METHOD} does not create movement. Nothing to plot here')
        return

    # This method creates a visualization of a random path an agent takes
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    df = HumanMobilityModel.df
    active_nodes = df[df.activity_start_min <= t0]
    active_nodes = active_nodes[active_nodes.activity_end_min >= tend]
    an = active_nodes.p_id.values[0]

    HumanMobilityModel.Location.plot_location(fig, ax)

    xs, ys = HumanMobilityModel.nodes[an].x, HumanMobilityModel.nodes[an].y
    ax.scatter(xs[t0: tend], ys[t0: tend], s=3, alpha=1)
    xs = xs[t0: tend]
    #print(np.where(np.isnan(xs)))
    
    plt.savefig(f'./plots/human_mobility/{HumanMobilityModel.Location.loc_id}_{HumanMobilityModel.METHOD}_interpolation_test_singular.png')


def interpolation_test(HumanMobilityModel, t0, tend):
    if HumanMobilityModel.METHOD in ['baseline', 'random', 'clique']:
        print(f'{HumanMobilityModel.METHOD} does not create movement. Nothing to plot here')
        return
    
    # This method creates a visualization of a random path multiple agents take
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    df = HumanMobilityModel.df
    active_nodes = df[df.activity_start_min <= t0]
    active_nodes = active_nodes[active_nodes.activity_end_min >= tend]
    active_nodes = active_nodes.p_id.values[:9]

    for ax, an in zip(axs.flatten(), active_nodes):
        HumanMobilityModel.Location.plot_location(fig, ax)
        xs, ys = HumanMobilityModel.nodes[an].x, HumanMobilityModel.nodes[an].y
        ax.scatter(xs[t0: tend], ys[t0: tend], s=1, alpha=.3)
    
    plt.savefig(f'./plots/human_mobility/{HumanMobilityModel.Location.loc_id}_{HumanMobilityModel.METHOD}_interpolation_test.png')


if __name__=='__main__':
    # Simulation walkthrough for a single location
    # Load data as a pandas DataFrame
    df_base = pd.read_parquet('./VF_data/rns_data_2.parquet')[['p_id', 'activity_start_min', 'loc_id_end', 'activity_name_mct', 'activity_end_min']]
    df_base = df_base.astype({'activity_start_min': 'uint32', 'activity_end_min': 'uint32'})  # Increase memory for simulation in seconds

    # Set simualtion time, for this example we simulate over the entire time range from the TAPAS data
    t_start, t_end = df_base.activity_start_min.min(), df_base.activity_end_min.max()

    # Select a location
    # group by location and sort by size (number of visitors during simulated day)
    locations = df_base.groupby('loc_id_end').size().sort_values(ascending=False).index.values
    # Some example locations, that where used in our recent paper
    loc1018 = df_base[df_base.loc_id_end == locations[1018]]
    loc1003 = df_base[df_base.loc_id_end == locations[1003]]
    loc1015 = df_base[df_base.loc_id_end == locations[1015]]
    loc2101 = df_base[df_base.loc_id_end == locations[2101]]

    # Start simulation
    # Build Location
    Loc = Location(1015, 10, 10, 10, 10)
    # Build simulation class with one of the example locations
    HN = ContactNetwork(loc1018, Loc, t_start, t_end, time_scale_data=60)

    # (optional) set paraemters of simulation class
    HN.tlw_max_wt = 100

    # simulate movement, non movement based methods are simulated during network creation
    HN.make_movement(method='TLW')

    # animate a part of the simulation, supported only by movement based methods
    HN.animate_movement()

    # Make simulation to tacoma network
    HN.make_tacoma_network()

    # This will make a visualization of one ore many node trajectories, supported only by movement based methods
    interpolation_test_singular(HN, 500*60, 550*60)
    interpolation_test(HN, 500*60, 550*60)
    pass
