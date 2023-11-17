import numpy as np
import pandas as pd
import util
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.markers import MarkerStyle
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from scipy.spatial import KDTree
from scipy.sparse import triu
from scipy.spatial.distance import pdist, squareform
import random
from math import ceil, floor
from mobility import truncated_levy_walk, random_waypoint


def attractor_pdf(x, k):
    return (k - 1)/(1+x)**k

def attractor_icdf(k):
    # Returns a sample drawn from the attractor pdf
    p = random.uniform(0, 1)
    return (1-p)**(1/(1-k)) - 1

def generate_array(row):
    return np.arange(row['activity_start_min'], row['activity_end_min'] + 1)


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


class HumanMobilityNetwork:
    def __init__(self, df, Location, t_start, t_end, t_scale=1, n_scale=60):
        '''
        df: a pandas data frame where every row is trajectory concluding in loc_id_end, 
        with the following coloumns:
            p_id: a unique identifier mapping to a individual
            activity_start_min/activity_end_min: start/end time of the activity in minutes passed to the beginning
            of the simulation
            activity_name_mct: describtion of the activity done at location loc_id_end
        
        Location: An instance of Location class defining important parameters for simulation

        t_start, t_end: start/end time of the simulation. Unit: minutes

        t_scale: Time scale of the simulation is 1 second i.e. one time step in the simulation happens every 1 second. The t_scale parameter is relevant if you want to return the dynamic network. 
        One time step in the dynamic network is aggregated from the simulation. An edge occurs if agents i,j met at least once during t_scale time steps in the simulation.
        n_scale: The number of seconds one time unit in the df represents
        '''
        self.Location = Location
        self.df = df
        print('start')
        # FOR TESTING!!!
        # self.df = self.df.head(100)

        # map p_id to index like
        util.map_pid(self.df)

        # print(self.df)

        # Setting parameters
        self.t_scale = t_scale
        self.n_scale = n_scale
        self.unique_nodes = self.df.p_id.unique()
        self.t_start = int(t_start * self.n_scale)
        self.t_end = ceil(t_end * self.n_scale)
        self.df.activity_start_min = (self.df.activity_start_min * self.n_scale).round().astype('int')
        self.df.activity_end_min = (self.df.activity_end_min * self.n_scale).round().astype('int')

        # Initialize nodes this will be done on simulation level later
        self.nodes = [Node(node, self.t_end) for node in np.arange(0, self.df.p_id.max() + 1)]

        # Model parameters
        self.k = 1.75 # Strenght of the attractor must be greater than 1
        self.STEPS_pause_time = 1.2 # shape parameter of Pareto PDF, this value has to be set interms of t_scal, it doesn't scale linear
        # self.STEPS_pause_time_inv = 1 / self.STEPS_pause_time
        self.v_STEPS_min, self.v_STEPS_max = .83, 3.2 # interval for uniform distribution to pick travel speed between zones from
        self.v_RWP_min, self.v_RWP_max = .1, 1. # interval for uniform distribution to pick travel speed between waypoints from
        self.RWP_tpause_min, self.RWP_tpause_max = 0, 5 # ranges to pick waiting time in waypoint from
        self.tlw_max_wt = 100

    def RWP(self, NODE, SPACE, t_start_RWP, t_end_RWP, x, y):
        t = t_start_RWP
        xp, yp = x, y

        while t < t_end_RWP:
            # pause
            tpause = random.randint(self.RWP_tpause_min, self.RWP_tpause_max)

            
            if t + tpause > t_end_RWP:
                tpause = t_end_RWP - t

            NODE.x[t: t + tpause] = xp
            NODE.y[t: t + tpause] = yp

            # choose new waypoint
            x, y = SPACE.get_random_coords()

            # travel to new waypoint
            distance = ((x - xp)**2 + (y - yp)**2)**.5
            travel_speed = random.uniform(self.v_RWP_min, self.v_RWP_max)
            travel_time = round(distance / travel_speed)

            if travel_time == 0:
                # teleport node to new location
                NODE.x[t + tpause] = x
                NODE.y[t + tpause] = y

                # Update time
                t = t + travel_time + tpause
                xp, yp = x, y
                continue

            if travel_time + tpause + t > t_end_RWP:
                # If not enough time is left calculate only part of trajectory
                vx = (x - xp) / travel_time
                vy = (y - yp) / travel_time
                travel_time = t_end_RWP - (tpause + t)
                x = xp + travel_time * vx # travel_speed should be the vx component here
                y = yp + travel_time * vy # travel_speed should be the vy component here
            
            x_interpolated = np.linspace(xp, x, travel_time)
            y_interpolated = np.linspace(yp, y, travel_time)
            NODE.x[t + tpause: t + tpause + travel_time] = x_interpolated
            NODE.y[t + tpause: t + tpause + travel_time] = y_interpolated

            # Update time
            t = t + travel_time + tpause
            xp, yp = x, y

        return xp, yp

    def RWP_main(self, nagents, dim):
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
            tpause = random.randint(20, 30)
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
        tpause = round(self.n_scale * random.paretovariate(self.STEPS_pause_time))

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
            tpause = round(self.n_scale *random.paretovariate(self.STEPS_pause_time))

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

            # RWP
            # xp, yp = self.RWP(NODE, self.Location.spaces[z], t + travel_time, t + travel_time + tpause, x, y)
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

    def get_TLW_positions(self, grp, pos):
        indices = np.concatenate(grp.to_numpy())
        self.nodes[grp.name].x[indices] = pos[indices, grp.name, 0]
        self.nodes[grp.name].y[indices] = pos[indices, grp.name, 1]

    def make_movement(self, method):
        self.METHOD = method

        if method == 'STEPS':
            # Assign default zones to all pid
            self.Location.get_distance_matrix()
            default_zones_map = dict(zip(self.unique_nodes, np.random.randint(0, self.Location.N_zones, self.unique_nodes.shape[0])))
            self.df['default_zone'] = self.df.p_id.map(default_zones_map)
            # print(self.df.activity_end_min.max())
            self.df.apply(self.STEPS, axis=1)
        
        if method == 'STEPS_with_RWP':
            # get RWP positions
            dim = (self.Location.space_dim_x, self.Location.space_dim_y)
            pos = self.RWP_main(len(self.unique_nodes), dim)
            print('finished RWP')
            # Assign default zones to all pid
            self.Location.get_distance_matrix()
            default_zones_map = dict(zip(self.unique_nodes, np.random.randint(0, self.Location.N_zones, self.unique_nodes.shape[0])))
            self.df['default_zone'] = self.df.p_id.map(default_zones_map)
            # print(self.df.activity_end_min.max())
            self.df.apply(self.STEPS_with_RWP, pos=pos, axis=1)

        
        if method == 'TLW':
            pos = self.TLW(len(self.unique_nodes))
            print(pos.shape)

            result = self.df.apply(generate_array, axis=1)
            result.groupby(self.df['p_id']).apply(self.get_TLW_positions, pos=pos)
        
        if method == 'RWP':
            dim = (self.Location.space_dim_x * self.Location.spaces_x, self.Location.space_dim_y * self.Location.spaces_x)
            pos = self.RWP_main(len(self.unique_nodes), dim)
            print(pos.shape)

            result = self.df.apply(generate_array, axis=1)
            result.groupby(self.df['p_id']).apply(self.get_TLW_positions, pos=pos)
    
    def get_spaces(self):
        X = np.array([node.x for node in self.nodes])
        Y = np.array([node.y for node in self.nodes])

        X_bounds = np.array([((i - 1)*self.Location.space_dim_x, i*self.Location.space_dim_x) for i in range(1, self.Location.spaces_x + 1)])
        Y_bounds = np.array([((i - 1)*self.Location.space_dim_y, i*self.Location.space_dim_y) for i in range(1, self.Location.spaces_y + 1)])
        # Y_bounds[0, 0] = Y_bounds[0, 0] - 1
        # print(Y_bounds)
        # X_bounds[0, 0] = X_bounds[0, 0] - 1
        # Reshape x and y positions to have a third dimension for intervals
        X = X[:, :, np.newaxis]
        Y = Y[:, :, np.newaxis]

        # Use broadcasting to check which intervals each agent position belongs to
        in_x_space = (X > X_bounds[:, 0]) & (X <= X_bounds[:, 1])
        in_y_space = (Y > Y_bounds[:, 0]) & (Y <= Y_bounds[:, 1])

        # Find Trues
        agent, time, space_x = np.where(in_x_space)
        _, _, space_y = np.where(in_y_space)

        '''for i, (tx, ty, ax, ay, sx, sy) in enumerate(zip(timex, timey, agentx, agenty, space_x, space_y)):
            if sx == 9:
                print(i, tx, ty, ax, ay, sx, sy, X[ax, tx], Y[ay, ty])
            if tx != ty:
                print(i-1, timex[i-1], timey[i-1], agentx[i-1], agenty[i-1], space_x[i-1], space_y[i-1], X[agentx[i-1], timex[i-1]], Y[agenty[i-1], timey[i-1]])
                print(i, tx, ty, ax, ay, sx, sy, X[ax, tx], Y[ay, ty])
                break'''

        space = space_x + self.Location.space_dim_y * space_y

        return time, agent, space
    
    def make_network(self, pos):
        posTree = KDTree(pos)
        relevant_distances = triu(posTree.sparse_distance_matrix(posTree, max_distance=5, p=2), k=1)

        Aind, Bind, dist = relevant_distances.row, relevant_distances.col, relevant_distances.data
        nodeA = pos[Aind]
        nodeB = pos[Bind]

        segments = np.stack((nodeA, nodeB), axis=1)
        return segments, dist
    
    def make_tacoma_network(self):
        pass

    def animate_movement(self):
        fig, ax = plt.subplots(figsize=(9, 9))
        self.Location.plot_location(fig, ax)

        X = np.array([node.x for node in self.nodes])
        Y = np.array([node.y for node in self.nodes])
        pos = np.array((X, Y)).T
        norm = Normalize(vmin=0, vmax=5)
        segments, dist = self.make_network(pos[0])
        lc = LineCollection(segments, cmap='Reds_r', norm=norm, linewidth=2)
        lc.set_array(dist)
        norm = Normalize(vmin=0, vmax=5)
        ax.add_collection(lc)
        scat = ax.scatter(pos[:, 0], pos[:, 1], c='grey')
        cbar = plt.colorbar(lc, ax=ax)
        cbar.set_label('distance [m]')

        def animate(frame):
            if frame % 10 == 0:
                print(f'{frame}/{self.t_end - self.t_start}')

            # update nodes
            x = X[:, frame]
            y = Y[:, frame]
            data = np.stack([x, y]).T
            scat.set_offsets(data)
            ax.set_title(f'{self.METHOD}, TU: {int(frame)}, 5xTU/s')

            # update edges
            segments, dist = self.make_network(pos[frame])
            lc.set_segments(segments)
            lc.set_array(dist)

        anim = FuncAnimation(fig, animate, range(round((self.t_end - self.t_start)/2), round((self.t_end - self.t_start)/2 + 300)), interval=200)
        anim.save(f'./plots/human_mobility/{self.Location.loc_id}_{self.METHOD}_animation_test.gif')


def interpolation_test_singular(HumanMobilityModel, t0, tend, method):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    df = HumanMobilityModel.df
    active_nodes = df[df.activity_start_min <= t0]
    active_nodes = active_nodes[active_nodes.activity_end_min >= tend]
    an = active_nodes.p_id.values[0]

    HumanMobilityModel.Location.plot_location(fig, ax)

    xs, ys = HumanMobilityModel.nodes[an].x, HumanMobilityModel.nodes[an].y
    ax.scatter(xs[t0: tend], ys[t0: tend], s=3, alpha=1)
    xs = xs[t0: tend]
    print(np.where(np.isnan(xs)))
    
    plt.savefig(f'./plots/human_mobility/{HumanMobilityModel.Location.loc_id}_{method}_interpolation_test_singular.png')


def interpolation_test(HumanMobilityModel, t0, tend, method):
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    df = HumanMobilityModel.df
    active_nodes = df[df.activity_start_min <= t0]
    active_nodes = active_nodes[active_nodes.activity_end_min >= tend]
    active_nodes = active_nodes.p_id.values[:9]

    for ax, an in zip(axs.flatten(), active_nodes):
        HumanMobilityModel.Location.plot_location(fig, ax)
        xs, ys = HumanMobilityModel.nodes[an].x, HumanMobilityModel.nodes[an].y
        ax.scatter(xs[t0: tend], ys[t0: tend], s=1, alpha=.3)
    
    plt.savefig(f'./plots/human_mobility/{HumanMobilityModel.Location.loc_id}_{method}_interpolation_test.png')



if __name__=='__main__':
    df_base = pd.read_parquet('./VF_data/rns_data_2.parquet')[['p_id', 'activity_start_min', 'loc_id_end', 'activity_name_mct', 'activity_end_min']]
    df_base = df_base.astype({'activity_start_min': 'uint32', 'activity_end_min': 'uint32'})
    t_start, t_end = df_base.activity_start_min.min(), df_base.activity_end_min.max()

    # location groups
    locations = df_base.groupby('loc_id_end').size().sort_values(ascending=False).index.values

    # Some example locations
    loc1018 = df_base[df_base.loc_id_end == locations[1018]]
    loc1003 = df_base[df_base.loc_id_end == locations[1003]]
    loc1015 = df_base[df_base.loc_id_end == locations[1015]]
    loc2101 = df_base[df_base.loc_id_end == locations[2101]]

    Loc = Location(1015, 10, 10, 10, 10)
    HN = HumanMobilityNetwork(loc1015, Loc, t_start, t_end, 1)
    HN.make_movement(method='STEPS_with_RWP')
    HN.animate_movement()


    '''t_scale = 1
    for method in ['TLW', 'RWP', 'STEPS', 'STEPS_with_RWP']:
        print(method)
        Loc = Location(1015, 10, 10, 10, 10)
        HN = HumanMobilityNetwork(loc1015, Loc, t_start, t_end, t_scale)
        HN.make_movement(method=method)

        interpolation_test_singular(HN, round(500*60/t_scale), round(600*60/t_scale), method) 
        interpolation_test(HN, round(500*60/t_scale), round(800*60/t_scale), method) 
        HN.animate_movement()  '''



    

    pass
