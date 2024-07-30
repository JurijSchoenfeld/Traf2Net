import numpy as np
import pandas as pd
import util
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.markers import MarkerStyle
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from scipy.spatial import KDTree
from scipy.sparse import triu, coo_array, find
from scipy.spatial.distance import pdist, squareform
import random
from math import ceil, floor, sqrt
from mobility import truncated_levy_walk, random_waypoint
import tacoma as tc
import clique
from tqdm import tqdm
from datetime import datetime
import os
import ast
import sys
from io import StringIO
import time


class DummyOutput:
    def write(self, text):
        pass


def silent_print(func):
    def wrapper(*args, **kwargs):
        # Redirect stdout to a dummy object
        sys.stdout = StringIO()
        # Call the function
        result = func(*args, **kwargs)
        # Restore stdout
        sys.stdout = sys.__stdout__
        return result
    return wrapper


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


def build_location(df, loc_id, loc_type, N_pps=20, school_space=3., capacity=None):
    # Source: Predicting the effects of COVID-19 related interventions in urban settings by combining activity-based modelling, agent-based simulation, and mobile phone data - Müller et al. 2021
    min_area_type = {'highschool': 2.0, 'office': 10.0, 'primaryschool': 2.0, 'supermarked': 10.0, 'restaurant': 1.25}

    # Get max visitors at full capacity
    t0 = df.activity_start_min.min()
    tmax = df.activity_end_min.max()

    if not capacity:
        number_visitors = np.zeros(shape=tmax-t0)
        for _, row in df.iterrows():
            number_visitors[row.activity_start_min - t0: row.activity_end_min - t0] += 1

        N_v = np.max(number_visitors)
    else:
        N_v = capacity

    # Get area
    A = N_v * min_area_type[loc_type]

    # If Npps is not provided create just a single space
    if not N_pps:
        N_pps = N_v
    
    # Number of zones in x/y direction
    Z_xy = ceil(sqrt(N_v/N_pps))

    # width/height of single Zone
    dim_xy = sqrt(A) / Z_xy

    return Location(loc_id, Z_xy, Z_xy, dim_xy, dim_xy)


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
        # self.df = self.df.head(10)

        # map p_id to index like
        util.map_pid(self.df)

        # Setting parameters
        self.time_scale_data = time_scale_data
        self.unique_nodes = self.df.p_id.unique()
        self.t_start = int(t_start * self.time_scale_data) # convert to seconds
        self.t_end = int(t_end * self.time_scale_data) # convert to seconds
        self.df.activity_start_min = (self.df.activity_start_min * self.time_scale_data).round().astype('int') # convert to seconds
        self.df.activity_end_min = (self.df.activity_end_min * self.time_scale_data).round().astype('int') # convert to seconds

        # Initialize nodes this will be done on simulation level later on
        self.nodes = [Node(node, self.t_end - self.t_start) for node in np.arange(0, self.df.p_id.max() + 1)]

        # Model parameters
        self.phi_fov = 2*np.pi/3  # sensor field of view in degrees
        self.phi_fov_over_2 = self.phi_fov / 2
        self.face_to_face_sampling_probability = (self.phi_fov/360)**2
        # STEPS
        self.k = 1.82 # Strenght of the attractor must be greater than 1
        self.STEPS_pause_min, self.STEPS_pause_max = 20, 30 # time spend in zone
        self.v_STEPS_min, self.v_STEPS_max = .83, 3.6 # interval for uniform distribution to pick travel speed between zones from
        # self.STEPS_paras = {'method': 'STEPS', 'k': self.k, 'STEPS_pause_min': self.STEPS_pause_min, 'STEPS_pause_max': self.STEPS_pause_max}
        
        # RWP
        self.v_RWP_min, self.v_RWP_max = .1, 1. # interval for uniform distribution to pick travel speed between waypoints from
        self.RWP_WT_MAX = 5  # max waiting time
        # self.RWP_paras = {'method': 'RWP', 'v_RWP_min': self.v_RWP_min, 'v_RWP_max': self.v_RWP_max, 'RWP_WT_MAX': self.RWP_WT_MAX}

        # STEPS with RWP
        # self.STEPS_with_RWP_paras = {**self.STEPS_paras, **self.RWP_paras} # merge both parameter dictionaries
        # self.STEPS_with_RWP_paras['method'] = 'STEPS_with_RWP'
        
        # TLW
        self.TLW_WT_MAX = 100  # maximum value of the waiting time distribution. Default is 100
        self.TLW_WT_EXP = -1.8  # exponent of the waiting time distribution. Default is -1.8
        self.FL_MAX = 50  # maximum value of the flight length distribution. Default is 50
        self.FL_EXP = -2.6  # exponent of the flight length distribution. Default is -2.6
        # self.TLW_paras = {'method': 'TLW', 'TLW_WT_MAX': self.TLW_WT_MAX, 'TLW_WT_EXP': self.TLW_WT_EXP, 'FL_MAX': self.FL_MAX, 'FL_EXP': self.FL_EXP}

        # STEPS and STEPS_with_RWP pareto
        self.STEPS_pareto = .8

        # Baseline
        self.min_contact_duration = None
        # elf.baseline_paras = {'method': 'baseline', 'min_contact_duration': self.min_contact_duration}

        # Random
        self.p_add, self.pareto_shape = .03, 3.
        # self.random_paras = {'method': 'random', 'p_add': self.p_add, 'pareto_shape': self.pareto_shape}

        # Clique
        self.N_PeoplePerSpace, self.p_space_change, self.mean, self.sigma = len(self.nodes) / self.Location.N_zones, 1/100, 10, 5
        # self.clique_paras = {'method': 'clique', 'N_people_per_space': self.N_PeoplePerSpace, 'p_space_change': self.p_space_change, 'mean': self.mean, 'sigma': self.sigma}

        # print('Initialized contact network model')


    def change_fov(self, new_fov):
        self.phi_fov = new_fov
        self.phi_fov_over_2 = self.phi_fov / 2
        self.face_to_face_sampling_probability = (self.phi_fov/360)**2


    def RWP_main(self, nagents, dim):
        # Returns RWP positions for n-agents moving in a room with dimensions dim
        rwp = random_waypoint(nagents, dimensions=dim, wt_max=self.RWP_WT_MAX, velocity=(self.v_RWP_min, self.v_RWP_max))
        pos = np.array([next(rwp).copy() for _ in range(self.t_end - self.t_start)])
        return pos


    def STEPS_with_RWP_pareto_func(self, row, pos):
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
            tpause = ceil(random.paretovariate(self.STEPS_pareto))  # this bit is not in line with the STEPS paper, instead it uses the probability distribution from their Matlab code 
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
            travel_time = 1 + ceil(distance / travel_speed)

            if travel_time <= 1:
                travel_time = 1
                x_interpolated = np.array([x])
                y_interpolated = np.array([y])
            else:
                x_interpolated = np.linspace(xp, x, travel_time)
                y_interpolated = np.linspace(yp, y, travel_time)

            if travel_time + t >= TE:
                # TODO: Find a smarter way to end trajectory in the future
                break

            NODE.x[t: t + travel_time] = x_interpolated
            NODE.y[t: t + travel_time] = y_interpolated

            if t + travel_time + tpause > TE:
                tail = TE - (t + travel_time + tpause)
                tpause = TE - (t + travel_time)
                rwp_pos_x = rwp_pos_x[:tail]
                rwp_pos_y = rwp_pos_y[:tail]

            # RWP
            NODE.x[t + travel_time: t + travel_time + tpause] = rwp_pos_x
            NODE.y[t + travel_time: t + travel_time + tpause] = rwp_pos_y
            
            # Update
            t = t + travel_time + tpause
            xp, yp = rwp_pos_x[-1], rwp_pos_y[-1]


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
            travel_time = 1 + ceil(distance / travel_speed)  # travel for at least 2 seconds

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


    def STEPS_pareto_func(self, row):
        TS, TE = row.activity_start_min, row.activity_end_min
        Z0 = row.default_zone
        NODE = self.nodes[row.p_id]
        
        # Initialize
        t = TS
        tpause = ceil(random.paretovariate(self.STEPS_pareto))  # this bit is not in line with the STEPS paper, instead it uses the probability distribution from their Matlab code 

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
            tpause = ceil(random.paretovariate(self.STEPS_pareto))  # this bit is not in line with the STEPS paper, instead it uses the probability distribution from their Matlab code 

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

            travel_time = 1 + ceil(distance / travel_speed)

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
        tlw = truncated_levy_walk(nagents, dimensions=dim, FL_EXP=self.FL_EXP, FL_MAX=self.FL_MAX, WT_EXP=self.TLW_WT_EXP, WT_MAX=self.TLW_WT_MAX)
        pos = np.array([next(tlw).copy() for _ in range(self.t_end - self.t_start)])
        return pos


    def baseline(self):
        start_times, end_times = self.df.activity_start_min.values, self.df.activity_end_min.values
        event_ids = self.df.p_id.values
        # loc_id_end = self.df.loc_id_end.values[0]
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
                        'end_of_contact': overlap_end[rows, cols].astype('int')})#,
                        #'loc_id': np.repeat(loc_id_end, len(p_A)).astype('int32')})
        
        # Calculate contact durations
        df_contacts['contact_duration'] = df_contacts.end_of_contact - df_contacts.start_of_contact

        # (optional) drop contacts below specified contact duration
        if self.min_contact_duration:
            df_contacts = df_contacts[df_contacts.contact_duration >= self.min_contact_duration]
        
        return df_contacts


    def random(self, contact):
        tstart, tend = contact.start_of_contact, contact.end_of_contact
    
        # Select contacts randomly
        selected_contact = np.full(tend - tstart, False) 
        # Random start times
        start = np.where(np.random.rand(tend - tstart) < self.p_add)[0]
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


    #@silent_print
    def make_movement(self, method):
        print('Start making movement')
        print('Parameters:')
        self.METHOD = method
        
        if method == 'baseline':
            self.paras = {'method': 'baseline', 'min_contact_duration': self.min_contact_duration}
            print(self.paras)

        elif method == 'random':
            self.paras = {'method': 'random', 'p_add': self.p_add, 'pareto_shape': self.pareto_shape}
            print(self.paras)

        elif method == 'clique':
            self.paras = {'method': 'clique', 'N_people_per_space': self.N_PeoplePerSpace, 'p_space_change': self.p_space_change, 'mean': self.mean, 'sigma': self.sigma}
            print(self.paras)

        elif method == 'clique_with_random':
            self.paras = {'method': 'clique_with_random', 'N_people_per_space': self.N_PeoplePerSpace, 'p_space_change': self.p_space_change, 'mean': self.mean, 'sigma': self.sigma, 'p_add': self.p_add, 'pareto_shape': self.pareto_shape}
            print(self.paras)

        elif method == 'STEPS':
            self.paras = {'method': 'STEPS', 'k': self.k, 'STEPS_pause_min': self.STEPS_pause_min, 'STEPS_pause_max': self.STEPS_pause_max, 'v_STEPS_min': self.v_STEPS_min, 'v_STEPS_max': self.v_STEPS_max, 'Npps': self.N_PeoplePerSpace}
            print(self.paras)
            # Assign default zones to all pid
            self.Location.get_distance_matrix()
            default_zones_map = dict(zip(self.unique_nodes, np.random.randint(0, self.Location.N_zones, self.unique_nodes.shape[0])))
            self.df['default_zone'] = self.df.p_id.map(default_zones_map)
            self.df.apply(self.STEPS, axis=1)
        
        elif method == 'STEPS_with_RWP':
            self.paras = {'method': 'STEPS_with_RWP', 'k': self.k, 'STEPS_pause_min': self.STEPS_pause_min, 'STEPS_pause_max': self.STEPS_pause_max, 'v_STEPS_min': self.v_STEPS_min, 'v_STEPS_max': self.v_STEPS_max, 'v_RWP_min': self.v_RWP_min, 'v_RWP_max': self.v_RWP_max, 'RWP_WT_MAX': self.RWP_WT_MAX, 'Npps': self.N_PeoplePerSpace}
            print(self.paras)
            # get RWP positions
            dim = (self.Location.space_dim_x, self.Location.space_dim_y)
            pos = self.RWP_main(len(self.unique_nodes), dim=dim)
        
            # Assign default zones to all pid
            self.Location.get_distance_matrix()
            self.default_zones_map = dict(zip(self.unique_nodes, np.random.randint(0, self.Location.N_zones, self.unique_nodes.shape[0])))
            self.df['default_zone'] = self.df.p_id.map(self.default_zones_map)
            # get STEPS positions
            self.df.apply(self.STEPS_with_RWP, pos=pos, axis=1)
        
        elif method == 'TLW':
            # TLW
            self.paras = {'method': 'TLW', 'TLW_WT_MAX': self.TLW_WT_MAX, 'TLW_WT_EXP': self.TLW_WT_EXP, 'FL_MAX': self.FL_MAX, 'FL_EXP': self.FL_EXP}
            print(self.paras)
            pos = self.TLW(len(self.unique_nodes))
            result = self.df.apply(util.generate_array, axis=1)
            result.groupby(self.df['p_id']).apply(self.get_positions, pos=pos)
        
        elif method == 'RWP':
            # RWP
            self.paras = {'method': 'RWP', 'v_RWP_min': self.v_RWP_min, 'v_RWP_max': self.v_RWP_max, 'RWP_WT_MAX': self.RWP_WT_MAX}
            print(self.paras)
            dim = (self.Location.space_dim_x * self.Location.spaces_x, self.Location.space_dim_y * self.Location.spaces_x)
            pos = self.RWP_main(len(self.unique_nodes), dim)

            result = self.df.apply(util.generate_array, axis=1)
            result.groupby(self.df['p_id']).apply(self.get_positions, pos=pos)

        elif method == 'STEPS_pareto':
            self.paras = {'method': 'STEPS_pareto', 'k': self.k, 'STEPS_pareto': self.STEPS_pareto, 'v_STEPS_min': self.v_STEPS_min, 'v_STEPS_max': self.v_STEPS_max, 'Npps': self.N_PeoplePerSpace, 'STEPS_pareto': self.STEPS_pareto, 'Npps': self.N_PeoplePerSpace}
            print(self.paras)
            # Assign default zones to all pid
            self.Location.get_distance_matrix()
            self.default_zones_map = dict(zip(self.unique_nodes, np.random.randint(0, self.Location.N_zones, self.unique_nodes.shape[0])))
            self.df['default_zone'] = self.df.p_id.map(self.default_zones_map)
            self.df.apply(self.STEPS_pareto_func, axis=1)
        
        elif method == 'STEPS_with_RWP_pareto':
            self.paras = {'method': 'STEPS_with_RWP_pareto', 'k': self.k, 'STEPS_pareto': self.STEPS_pareto, 'v_STEPS_min': self.v_STEPS_min, 'v_STEPS_max': self.v_STEPS_max, 'v_RWP_min': self.v_RWP_min, 'v_RWP_max': self.v_RWP_max, 'RWP_WT_MAX': self.RWP_WT_MAX, 'STEPS_pareto': self.STEPS_pareto, 'Npps': self.N_PeoplePerSpace}
            print(self.paras)
            # get RWP positions
            dim = (self.Location.space_dim_x, self.Location.space_dim_y)
            pos = self.RWP_main(len(self.unique_nodes), dim=dim)
        
            # Assign default zones to all pid
            self.Location.get_distance_matrix()
            self.default_zones_map = dict(zip(self.unique_nodes, np.random.randint(0, self.Location.N_zones, self.unique_nodes.shape[0])))
            self.df['default_zone'] = self.df.p_id.map(self.default_zones_map)
            # get STEPS positions
            self.df.apply(self.STEPS_with_RWP_pareto_func, pos=pos, axis=1)


    def network_animation(self, pos, UV, t):
        # Make network and retun segments, dist for animation
        posTree = KDTree(pos)
        distances = triu(posTree.sparse_distance_matrix(posTree, max_distance=1.5, p=2), k=1)
        # Get contacts longer than min dist
        min_dist_mask = (distances.data >= 0.0)

        Aind, Bind, dist = distances.row[min_dist_mask], distances.col[min_dist_mask], distances.data[min_dist_mask]
        # Aind, Bind, dist = distances.row, distances.col, distances.data
        # Select contacts that are face to face
        R_AB = pos[Bind] - pos[Aind]
        UV_A, UV_B = UV[t, Aind], UV[t, Bind]
        UV_A_norm, UV_B_norm = np.linalg.norm(UV_A, ord=2, axis=1), np.linalg.norm(UV_B, ord=2, axis=1)
        R_AB_norm = np.linalg.norm(R_AB, ord=2, axis=1)

        phi_A = np.arccos(np.sum(UV_A * R_AB, axis=1) / (R_AB_norm * UV_A_norm))
        phi_B = np.arccos(np.sum(-UV_B * R_AB, axis=1) / (R_AB_norm * UV_B_norm))

        # Select Edges where nodes can see each other
        sight_mask = (phi_A <= self.phi_fov/2) & (phi_B <= self.phi_fov/2)
        Aind = Aind[sight_mask]
        Bind = Bind[sight_mask]


        nodeA = pos[Aind]
        nodeB = pos[Bind]

        segments = np.stack((nodeA, nodeB), axis=1)

        dist = dist[sight_mask]

        return segments, dist


    #@silent_print
    def tn_from_contacts(self, contacts, to):
        # Initilize tacoma temporal network from contacts generated by model
        tn = tc.edge_changes()
        tn.N = len(self.unique_nodes)
        # tmax, tmin = self.t_end, self.t_start
        Nt = ceil(24*60*60/20) + 1

        tn.t = list(range(Nt))
        tn.t0 = -1
        tn.tmax = Nt
        tn.time_unit = '20s'
        tn.notes = str(self.paras)

        # Initial edges
        edges_in, edges_out = [[] for _ in range(Nt)], [[] for _ in range(Nt)]
        # contacts_initial = contacts[contacts.start_of_contact == tmin]
        # contacts = contacts[contacts.start_of_contact != tmin]

        # Make other edges
        for _, c in contacts[['p_A', 'p_B', 'start_of_contact', 'end_of_contact']].iterrows():
            edges_in[c.start_of_contact + to].append([c.p_A, c.p_B])
            edges_out[c.end_of_contact + to].append([c.p_A, c.p_B])

        tn.edges_in = edges_in
        tn.edges_out = edges_out

        # Check for errors
        print('edge changes errors: ', tc.verify(tn))

        return tn


    #@silent_print
    def make_tacoma_network(self, min_dist=0.0, max_dist=1.5, time_resolution=20, export=False, temporal_offset=None):
        print('Start network construction')
        print(self.phi_fov)
        # Write model parameter to tacoma network
        self.paras['time_resolution'] = time_resolution
        self.paras['max_dist'] = max_dist
        self.paras['loc_id'] = self.Location.loc_id
        self.paras['spaces_xy'] = self.Location.spaces_x
        self.paras['space_dim_xy'] = self.Location.space_dim_x

        if not temporal_offset:
            temporal_offset = self.t_start
        
        if self.METHOD in ['RWP', 'TLW', 'STEPS', 'STEPS_with_RWP', 'STEPS_pareto', 'STEPS_with_RWP_pareto']:
            # Get node positions
            X = np.array([node.x for node in self.nodes])
            Y = np.array([node.y for node in self.nodes])
            all_pos = np.array((X, Y)).T

            # Get node direction of sight
            U = np.diff(X, axis=1)
            U = util.forward_fill_zeros(U)

            V = np.diff(Y, axis=1)
            V = util.forward_fill_zeros(V)

            # Add 0 to first time stamp
            UV = np.array((U, V)).T
            UV = np.insert(UV, 0, np.zeros((1, UV.shape[1], UV.shape[2])), axis=0)
            
            # Initiate tacoma dynamic network
            tn = tc.edge_lists()
            tn.notes = str(self.paras)
            tn.N = len(self.unique_nodes)
            
            Nt = ceil(24*60*60/time_resolution) + 1
            
            tn.t = list(range(Nt))
            tn.tmax = Nt
            tn.time_unit = f'{time_resolution}s'
            contacts = [[] for _ in range(Nt)]
            # relevant_distances = coo_array((tn.N, tn.N))
            relevant_contacts = set()
 
            for t, XY in enumerate(all_pos):
                posTree = KDTree(XY)
                # Add all contacts together
                # Get contacts shortes than max dist
                distances = triu(posTree.sparse_distance_matrix(posTree, max_distance=max_dist, p=2), k=1)
                # Get contacts longer than min dist
                min_dist_mask = (distances.data >= min_dist)
                Aind, Bind = distances.row[min_dist_mask], distances.col[min_dist_mask]

                # Select contacts that are face to face
                R_AB = XY[Bind] - XY[Aind]
                UV_A, UV_B = UV[t, Aind], UV[t, Bind]
                UV_A_norm, UV_B_norm = np.linalg.norm(UV_A, ord=2, axis=1), np.linalg.norm(UV_B, ord=2, axis=1)
                R_AB_norm = np.linalg.norm(R_AB, ord=2, axis=1)

                phi_A = np.arccos(np.sum(UV_A * R_AB, axis=1) / (R_AB_norm * UV_A_norm))
                phi_B = np.arccos(np.sum(-UV_B * R_AB, axis=1) / (R_AB_norm * UV_B_norm))

                # Select Edges where nodes can see each other
                sight_mask = (phi_A <= self.phi_fov_over_2) & (phi_B <= self.phi_fov_over_2)
                Aind = Aind[sight_mask]
                Bind = Bind[sight_mask]

                relevant_contacts.update(set(zip(Aind, Bind)))

                if (t + 1) % time_resolution == 0:
                    # Add to contact python list
                    contacts[int((t + 1)/time_resolution) + temporal_offset].extend(relevant_contacts)
                    relevant_contacts.clear()
                
                if t % 10000 == 0:
                    print(f'{t}/{self.t_end - self.t_start}')

            # Check for errors and convert to edge_changes
            tn.edges = contacts
            print('edge list errors: ', tc.verify(tn))

            tn = tc.convert(tn)
            print('edge changes errors: ', tc.verify(tn))
            self.tn = tn
            self.tn.time_unit = f'{time_resolution}s'

        elif self.METHOD == 'baseline':
            # Get contacts
            contacts = self.baseline()
            
            # Downscale contact time
            contacts = util.downscale_time_contacts(contacts, time_resolution)
            self.tn = self.tn_from_contacts(contacts, temporal_offset)
  

        elif self.METHOD == 'random':
            # Get all possible contacts
            contacts = self.baseline()

            # Select random contacts
            selected_contacts = contacts.apply(self.random, axis=1)
            selected_contacts = pd.concat(selected_contacts.to_list(), ignore_index=True) 
            selected_contacts = util.downscale_time_contacts(selected_contacts, time_resolution)      
            self.tn = self.tn_from_contacts(selected_contacts, temporal_offset)


        elif self.METHOD == 'clique':
            # Select contacts if nodes share space
            space_contacts = clique.get_contacts_spaces(self.df, self.N_PeoplePerSpace, self.p_space_change, self.mean, self.sigma)

            # Downscale contact time
            space_contacts = util.downscale_time_contacts(space_contacts, time_resolution)
            self.tn = self.tn_from_contacts(space_contacts, temporal_offset)

        elif self.METHOD == 'clique_with_random':
            # Select contacts if nodes share space
            space_contacts = clique.get_contacts_spaces(self.df, self.N_PeoplePerSpace, self.p_space_change, self.mean, self.sigma)

            # Select random contacts
            selected_contacts = space_contacts.apply(self.random, axis=1)
            selected_contacts = pd.concat(selected_contacts.to_list(), ignore_index=True)       

            # Downscale contact time
            space_contacts = util.downscale_time_contacts(selected_contacts, time_resolution)
            self.tn = self.tn_from_contacts(selected_contacts, temporal_offset)
        

        if export:
            # Get unique experiment id
            experiment_id = datetime.now().strftime("%Y%m%d%H%M%S")

            # Write network
            tc.write_json_taco(self.tn, f'./networks/{self.Location.loc_id}_{self.METHOD}_TU={time_resolution}_{experiment_id}.taco')

        return self.tn


    def load_tn(self, path, method):
        self.METHOD = method
        self.tn = tc.read_json_taco(path)
    

    def run_SIR(self, nruns, beta, gamma, ndays, normalize, save=True, plot=True):
        IS = run_SIR(self.tn, self.Location, self.METHOD, nruns, beta, gamma, ndays, normalize, save, plot)
    

    def find_epidemic_threshold(self, beta_range, gamma, nruns, ndays, normalize, save=True, plot=True):
        find_epidemic_threshold(self.tn, self.Location, self.METHOD, beta_range, gamma, nruns, ndays, normalize, save, plot)
        

    def animate_movement(self):
        print('Start animation')
        if self.METHOD in ['baseline', 'random', 'clique']:
            print(f'The specified method: {self.METHOD} does not support animation')
            return
        
        fig, ax = plt.subplots(figsize=(9, 9))
        plt.tight_layout()
        self.Location.plot_location(fig, ax)

        # Get node colors
        try:
            node_ids = np.array(list(self.default_zones_map.keys()))
            default_zones = np.array(list(self.default_zones_map.values()))
            node_colors = np.array(['grey' for _ in node_ids])
            selected_zone = random.randint(0, 2 * self.Location.spaces_x - 1)
            node_colors[default_zones == selected_zone] = 'blue'
            #print(node_colors)
            #print(selected_zone)
            #print(default_zones)

        except AttributeError:
            node_colors = 'grey'

        X = np.array([node.x for node in self.nodes])
        Y = np.array([node.y for node in self.nodes])
        pos = np.array((X, Y)).T
        # Get node direction of sight
        U = np.diff(X, axis=1)
        U = util.forward_fill_zeros(U)

        V = np.diff(Y, axis=1)
        V = util.forward_fill_zeros(V)

        # Add 0 to first time stamp
        UV = np.array((U, V)).T
        UV = np.insert(UV, 0, np.zeros((1, UV.shape[1], UV.shape[2])), axis=0)

        norm = Normalize(vmin=0.0, vmax=1.5)
        segments, dist = self.network_animation(pos[0], UV, 0)
        lc = LineCollection(segments, cmap='Reds_r', norm=norm, linewidth=2)
        lc.set_array(dist)

        sights = np.stack((pos[0], pos[0] + UV[0]), axis=1)
        lc_sights = LineCollection(sights, linewidth=2, color=node_colors)

        ax.add_collection(lc_sights)
        ax.add_collection(lc)
        
        #print('pos:', pos[0 ].shape)
        scat = ax.scatter(pos[0,:, 0], pos[0,:, 1], c=node_colors)
        #cbar = plt.colorbar(lc, ax=ax)
        #cbar.set_label('distance [m]')

        frame_start, frame_stop = round((self.t_end - self.t_start)/2), round((self.t_end - self.t_start)/2 + 600)  # animate 300 frames
        def animate(frame):
            if frame % 10 == 0:
                print(f'{frame}/{frame_stop}')

            # update nodes
            x = X[:, frame]
            y = Y[:, frame]
            active_nodes = np.sum(~np.isnan(x))
            data = np.stack([x, y]).T
            scat.set_offsets(data)
            # ax.set_title(f'N_V={active_nodes}, {self.METHOD}, Frame: {int(frame)}')

            segments, dist = self.network_animation(pos[frame], UV, frame)
            # update edges
            lc.set_segments(segments)
            lc.set_array(dist)

            # update sights
            sights = np.stack((data, data + UV[frame] / np.linalg.norm(UV[frame], ord=2, axis=1)[:, np.newaxis]), axis=1)
            lc_sights.set_segments(sights)

        anim = FuncAnimation(fig, animate, range(frame_start, frame_stop), interval=200)
        #anim.save(f'./plots/human_mobility/{self.Location.loc_id}_{self.METHOD}_animation_test.gif')
        anim.save(f'./final_paras_animation/{self.Location.loc_id}_{self.METHOD}_animation_test.gif')


def interpolation_test_singular(HumanMobilityModel, t0, tend):
    if HumanMobilityModel.METHOD in ['baseline', 'random', 'clique']:
        print(f'{HumanMobilityModel.METHOD} does not create movement. Nothing to plot here')
        return

    # This method creates a visualization of a random path an agent takes
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set(xlabel=r'$x$[m]', ylabel=r'$y$[m]', title=HumanMobilityModel.METHOD)
    df = HumanMobilityModel.df
    active_nodes = df[df.activity_start_min <= t0]
    active_nodes = active_nodes[active_nodes.activity_end_min >= tend]
    an = active_nodes.p_id.values[0]

    HumanMobilityModel.Location.plot_location(fig, ax)

    xs, ys = HumanMobilityModel.nodes[an].x, HumanMobilityModel.nodes[an].y
    ax.scatter(xs[t0: tend], ys[t0: tend], s=3, alpha=1)
    xs = xs[t0: tend]
    #print(np.where(np.isnan(xs)))
    #np.save(HumanMobilityModel.method,)
    
    plt.savefig(f'./final_paras_animation/{HumanMobilityModel.Location.loc_id}_{HumanMobilityModel.METHOD}_interpolation_test_singular.png')


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
    print(active_nodes)

    for ax, an in zip(axs.flatten(), active_nodes):
        HumanMobilityModel.Location.plot_location(fig, ax)
        xs, ys = HumanMobilityModel.nodes[an].x, HumanMobilityModel.nodes[an].y
        ax.scatter(xs[t0: tend], ys[t0: tend], s=1, alpha=.3)
    
    plt.savefig(f'./final_paras_animation/{HumanMobilityModel.Location.loc_id}_{HumanMobilityModel.METHOD}_interpolation_test.png')


def interpolation_test_all_methods(HumanMobilityModel, t0, tend, fig, ax):
    if HumanMobilityModel.METHOD in ['baseline', 'random', 'clique']:
        print(f'{HumanMobilityModel.METHOD} does not create movement. Nothing to plot here')
        return

    # This method creates a visualization of a random path an agent takes
    #fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    label = HumanMobilityModel.METHOD
    if label == 'STEPS_with_RWP_pareto':
        label = 'STEPS+RWP'
    elif label == 'STEPS_pareto':
        label = 'STEPS'

    if label == 'RWP' or label == 'STEPS':
        ax.set(ylabel=r'$y$ [m]')
    if label == 'STEPS' or label == 'STEPS+RWP':
        ax.set(xlabel=r'$x$ [m]')

    ax.set(title=label)
    df = HumanMobilityModel.df
    active_nodes = df[df.activity_start_min <= t0]
    active_nodes = active_nodes[active_nodes.activity_end_min >= tend]
    an = active_nodes.p_id.values[0]

    HumanMobilityModel.Location.plot_location(fig, ax)

    xs, ys = HumanMobilityModel.nodes[an].x, HumanMobilityModel.nodes[an].y
    ax.scatter(xs[t0: tend], ys[t0: tend], s=1, alpha=.3)
    xs = xs[t0: tend]
    #print(np.where(np.isnan(xs)))
    #np.save(HumanMobilityModel.method,)
    
    #plt.savefig(f'./final_paras_animation/{HumanMobilityModel.Location.loc_id}_{HumanMobilityModel.METHOD}_interpolation_test_singular.png')



def profile():
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

    Loc = Location(1018, 8, 8, 4.14578098794425, 4.14578098794425)
    # Build simulation class with one of the example locations
    HN = ContactNetwork(loc1018, Loc, t_start, t_end, time_scale_data=60)

    # (optional) set paraemters of simulation class
    HN.tlw_max_wt = 100

    # simulate movement, non movement based methods are simulated during network creation
    HN.make_movement(method='TLW')

    # animate a part of the simulation, supported only by movement based methods
    # HN.animate_movement()

    # Make simulation to tacoma network
    HN.make_tacoma_network(max_dist=2.0, time_resolution=60, export=True)


#@silent_print
def run_SIR(tn, Location, METHOD, nruns, beta, gamma, ndays, normalize, save=True, plot=True, save_path=None):
    # nruns (int): number of runs
    # beta (float): infection rate
    # gamma (foat): recovery rate
    # nadays (int): The simualtion is performed on a network that is looped ndays times

    # Calculate normalization factor and normalize network
    if normalize:
        _, _, m = tc.edge_counts(tn)
        norm_factor = np.sum(np.array(m))
    else:
        norm_factor = 1

    beta = beta / norm_factor
    print(beta)

    # Initilize time series of infection dynamic parameters
    time = np.arange(tn.tmax * ndays + 1)

    Is = []
    for _ in tqdm(range(nruns)):
        I = np.empty_like(time)
        I[:] = np.nan
        # Initialize epidemic class
        SIR = tc.SIR(tn.N, tn.tmax * ndays, beta, gamma)

        # Run 
        tc.gillespie_SIR(tn, SIR)
        
        # Extract results
        events = np.array(SIR.time, dtype='int')
        I[events] = SIR.I
        I = pd.Series(I).ffill().values.astype('uint16')
        Is.append(I)
    
    Is = np.array(Is)
    

    # Save results
    if save:
        if save_path:
            np.save(save_path, Is)
        else:
            np.save(f'./results/{Location.loc_id}_{METHOD}_beta={beta}_gamma={gamma}', Is)

    # Plot results
    if plot:
        time = time #/(60*24)
        Is = Is[:, :-1]
        time = time[:-1]
        I_mean, I_err_upper, I_err_lower = util.mean_with_errors(Is, tn.N)

        plt.plot(time, I_mean / tn.N, label='mean')
        plt.legend()

        plt.fill_between(time, I_err_lower / tn.N, I_err_upper / tn.N, alpha=.3)
        plt.xlabel('t in '+ tn.time_unit)
        plt.ylabel('relative number of infected')

        plt.savefig(f'./results/plots/{Location.loc_id}_{METHOD}_beta={beta}_gamma={gamma}_nruns={nruns}.png')
        plt.close()

    return Is


def find_epidemic_threshold(tn, Location, METHOD, beta_range, gamma, nruns, ndays, normalize=False, save=False, plot=True):
    # find the epidemic threshold in beta_range for a constant gamma
    # nruns is the number of runs for every (beta, gamma) combination

    # Number of nodes
    Nv = tn.N

    # Find epidemic threshold
    sigmas = []  # outbreak sizes
    sigma_errors = []
    for beta in beta_range:
        print(beta)
        Is = run_SIR(tn, Location, METHOD, nruns, beta, gamma, ndays, normalize=False, save=False, plot=True) / Nv
        maxima = np.max(Is, axis=1)
        sigmas.append(np.mean(maxima))
        sigma_errors.append(np.std(maxima))
    
    np.save(f'./results/epdicemic_threshold/data/{Location.loc_id}_{METHOD}_beta={beta_range[0]}_{beta_range[-1]}_gamme={gamma}_nruns={nruns}_days={ndays}_sigmas.npy', sigmas)
    np.save(f'./results/epdicemic_threshold/data/{Location.loc_id}_{METHOD}_beta={beta_range[0]}_{beta_range[-1]}_gamme={gamma}_nruns={nruns}_days={ndays}_errors.npy', sigma_errors)
    plt.errorbar(beta_range / gamma, sigmas, sigma_errors, capsize=5, linestyle='', marker='x')
    plt.xlabel(r'$\beta/\gamma$')
    plt.ylabel('relative infection size')
    plt.savefig(f'./results/epdicemic_threshold/plots/{Location.loc_id}_{METHOD}_beta={beta_range[0]}_{beta_range[-1]}_gamme={gamma}_nruns={nruns}_days={ndays}.png')
    plt.close()


def build_tapas_network_from_paras():
    df = pd.read_parquet('./VF_data/rns_uni_location.parquet')
    # Set simualtion time, for this example we simulate over the entire time range from the TAPAS data
    t_start, t_end = 0, 1440
    loc_id = df.loc_id_end.iloc[0]

    # Load tacoma network
    files = os.listdir('./networks/highschool/')
    network_paths = {file.split('_', maxsplit=1)[1].split('_TU')[0]: file for file in files}


    # Build Location
    for method in ['STEPS_with_RWP', 'STEPS', 'RWP', 'TLW']:
        Loc = build_location(df.copy(), loc_id, 'school', N_pps=20)

        # Load Simulation parameters
        tn = tc.load_json_taco('./networks/highschool/' + network_paths[method])
        paras = ast.literal_eval(tn.notes.split('\n')[0])
        CN = ContactNetwork(df.copy(), Loc, t_start, t_end, time_scale_data=60)

        for para, para_val in paras.items():
            try:
                setattr(CN, para, para_val)
            except ValueError:
                pass
    
        print(Loc.spaces_x, Loc.space_dim_x)
        # Start simulation
        CN.make_movement(method)
        CN.make_tacoma_network(max_dist=2, time_resolution=20, export=True)

        #print(CN.tn.edges_in)

        _, _, m = tc.edge_counts(CN.tn)
        plt.plot(CN.tn.t, m[:-1])
        plt.savefig(f'{method}_test_edges.png')


def fix_to_short_networks():
    methods = ['empirical', 'baseline', 'random', 'clique', 'RWP', 'TLW', 'STEPS', 'STEPS_with_RWP']
    for method in methods:
        # Load tacoma network
        files = os.listdir('./networks/highschool/')
        network_paths = {file.split('_', maxsplit=1)[1].split('_TU')[0]: file for file in files}
        tn = tc.load_json_taco('./networks/highschool/' + network_paths[method])
        # tn.tmax is 9h

        new_edges_in, new_edges_out = [[] for _ in range(1080)], [[] for _ in range(1080)]
        night = [[] for _ in range(1623)]

        new_edges_in.extend(tn.edges_in)
        new_edges_in.extend(night)

        new_edges_out.extend(tn.edges_out)
        new_edges_out.extend(night)

        t = list(np.arange(1, len(new_edges_in) + 1))
        new_edges_in[1079] = tn.edges_initial
        print(len(new_edges_in), len(new_edges_out), len(t))
        tn.t0, tn.tmax = 0, 4320 + 3
        tn.t = t
        tn.edges_in = new_edges_in
        tn.edges_out = new_edges_out
        tn.edges_initial = []


        tn = tc.convert(tn)
        edges = tn.edges[:2697]
        edges.extend(night)
        tn.edges = edges

        tn = tc.convert(tn) 

        print(tc.verify(tn))

        _, _, m = tc.edge_counts(tn)
        plt.plot(tn.t, m[:-1])
        plt.savefig(f'hs_{method}_test_edges.png')
        plt.close()

        tc.write_json_taco(tn, './networks/new_highschool/' + network_paths[method])


def tapas_networks(ntype, method):
    # Get SIR paras
    if ntype == 'office':
        capacity = 217
        path = './data_eval_split/InVS/f1_1970-01-01.parquet' 
        max_dist, min_dist, fov = 1.5, 0.0, 2.094395
        SIR_paras = .013, 20 / (7*24*60*60), 2170, 35  # beta, gamma, nruns, ndays
        SIR_path = './results/office_empirical_beta=0.013_gamma=3.306878306878307e-05.npy'

    elif ntype == 'highschool':
        capacity = 327
        path = './data_eval_split/highschool/f0_2013-12-03.parquet'
        max_dist, min_dist, fov = 1.5, 0.0, 2.094395
        SIR_paras = .007, 20 / (7*24*60*60), 3270, 35
        SIR_path = './results/highschool_empirical_beta=0.007_gamma=3.306878306878307e-05.npy'

    elif ntype == 'primaryschool':
        path = './data_eval_split/primaryschool/1970-01-01.parquet'
        capacity = 242
        max_dist, min_dist, fov = 1.5, 0.0, 2.094395
        SIR_paras = .0013, 20 / (7*24*60*60), 2420, 35
        SIR_path = './results/primaryschool_empirical_beta=0.0013_gamma=3.306878306878307e-05.npy'
    
    beta, gamma, nruns, ndays = SIR_paras

    # Get best parameters for given network type and method
    best_paras_dict = {'office': {'STEPS_pareto': {'k': 2.8809814385623684,
                                    'STEPS_pareto': 0.7676296683599062,
                                    'N_people_per_space': 24},
                                    'STEPS_with_RWP_pareto': {'k': 4.372608807670108,
                                    'STEPS_pareto': 0.7189126397421236,
                                    'N_people_per_space': 30,
                                    'RWP_WT_MAX': 1177},
                                    'clique_with_random': {'p_space_change': 0.026573954773857322,
                                    'mean': 271,
                                    'sigma': 28,
                                    'p_add': 0.038092639210002804,
                                    'pareto_shape': 1.592905968448615,
                                    'N_people_per_space': 2},
                                    'random': {'p_add': 0.00045426551451094066, 'pareto_shape': 3.08156770075869},
                                    'RWP': {'RWP_WT_MAX': 1797},
                                    'baseline': {'weight': 0.0005436035657689404},
                                    'TLW': {'TLW_WT_MAX': 3277,
                                    'TLW_WT_EXP': -0.28634300032673937,
                                    'FL_MAX': 49,
                                    'FL_EXP': -9.894483858860687}},
                        
                        'primaryschool': {'STEPS_pareto': {'k': 9.974003647535408,
                                            'STEPS_pareto': 0.6130675933667971,
                                            'N_people_per_space': 39},
                                            'STEPS_with_RWP_pareto': {'k': 8.202328300072152,
                                            'STEPS_pareto': 0.6998265484241938,
                                            'N_people_per_space': 5,
                                            'RWP_WT_MAX': 1133},
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
                                            'FL_EXP': -8.996563363396556}},
                        
                        'highschool':  {'STEPS_pareto': {'k': 4.3879605793130265,
                                        'STEPS_pareto': 0.4210183226476539,
                                        'N_people_per_space': 27},
                                        'STEPS_with_RWP_pareto': {'k': 4.86415873950932,
                                        'STEPS_pareto': 0.6189274019207999,
                                        'N_people_per_space': 2,
                                        'RWP_WT_MAX': 3324},
                                        'clique_with_random': {'p_space_change': 0.0562632459155923,
                                        'mean': 170,
                                        'sigma': 91,
                                        'p_add': 0.008016298535436495,
                                        'pareto_shape': 9.998054452780455,
                                        'N_people_per_space': 19},
                                        'random': {'p_add': 0.00034618042207039445,
                                        'pareto_shape': 7.061619041378239},
                                        'RWP': {'RWP_WT_MAX': 3599},
                                        'baseline': {'weight': 0.0002992849250132354},
                                        'TLW': {'TLW_WT_MAX': 2359,
                                        'TLW_WT_EXP': -0.5029329300743824,
                                        'FL_MAX': 80,
                                        'FL_EXP': -9.987833730779137}},
                        
                        'supermarked': {'STEPS_pareto': {'k': 3.6003296257524102,
                                        'STEPS_pareto': 0.46942379903047576,
                                        'N_people_per_space': 40},
                                        'RWP': {'RWP_WT_MAX': 610},
                                        'random': {'p_add': 0.009703836159227257, 'pareto_shape': 1.4881299058280244},
                                        'STEPS_with_RWP_pareto': {'k': 3.14003379565969,
                                        'STEPS_pareto': 1.6980257805820058,
                                        'N_people_per_space': 1,
                                        'RWP_WT_MAX': 3343},
                                        'TLW': {'TLW_WT_MAX': 1478,
                                        'TLW_WT_EXP': -6.288970594712486,
                                        'FL_MAX': 54,
                                        'FL_EXP': -9.763394702496802},
                                        'clique_with_random': {'p_space_change': 0.06713118209401989,
                                        'mean': 194,
                                        'sigma': 1,
                                        'p_add': 0.08438360458727655,
                                        'pareto_shape': 0.10588878039329785,
                                        'N_people_per_space': 39},
                                        'baseline': {'weight': 0.009970022805909147}}
                            
                            
                            
                            
                            }
    best_paras = best_paras_dict[ntype][method]
    
    # Load trajectories
    df_base = pd.read_parquet(f'./networks/{ntype}/tapas_{ntype}_trajectory.parquet')[['loc_id_end', 'p_id', 'activity_start_min', 'activity_duration_min']]
    df_base['activity_end_min'] = df_base.activity_start_min + df_base.activity_duration_min
    #t_start, t_end = df_base.activity_start_min.min(), df_base.activity_end_min.max()
    t_start, t_end = 0, 1440

    # Build location
    # Try to find Npps in best paras, if this fails set to 20 (in that case it should be of no importance to the method)
    try:
        Npps = best_paras['N_people_per_space']
    except KeyError:
        Npps = 20
    Loc = build_location(df_base, 'tapas_' + ntype, ntype, N_pps=Npps, capacity=None)  # Given capacities seem way to high -> estimate capas by computing how many nodes are present at max

    # Build network
    CN = ContactNetwork(df_base, Loc, t_start, t_end, time_scale_data=60)
    for para, para_value in best_paras.items():
        CN.__setattr__(para, para_value)
    
    CN.change_fov(fov)
    CN.make_movement(method)
    CN.make_tacoma_network(min_dist=min_dist, max_dist=max_dist, time_resolution=20, temporal_offset=0)

    print(CN.tn.N)

    # Run SIR
    run_SIR(CN.tn, Loc, method, nruns, beta, gamma, ndays, normalize=False, save=True, plot=False)


class MemilioContactNetwork:
    def __init__(self, ntype, N, C=None, loc_id=None):
        self.ntype = ntype
        self.N = N
        self.C = C
        self.loc_id = loc_id
        self.t_start, self.t_end = 0, 86_400

        # If no capacity is given, set capacity to the number of nodes
        if not self.C:
            self.C = self.N
        
        # If no loc_id is given loc_id as ntype_N_C
        if not self.loc_id:
            self.loc_id = f'{self.ntype}_{self.N}_{self.C}'

        # class variables that might be initialized later on
        self.df_activities = None
        self.Loc = None
        self.tn = None
        self.TU_aggregated = None


    def create_tapas_like_trajectories(self):
        # Create 24h activites for N nodes in location of ntype with capacity C
        # Make activity data
        loc_id_end = [self.loc_id] * self.N
        p_id = list(range(self.N))
        activity_start_min = [self.t_start] * self.N
        activity_end_min = [self.t_end] * self.N

        # Set modelparameters from paper
        best_paras_STEPS =  {
                        # Dicitionary with STEPS paras after tuning, changed Npps to actual value
                        'highschool':       {'k': 4.3879605793130265, 'STEPS_pareto': 0.4210183226476539, 'N_people_per_space': 20},
                        'primaryschool':    {'k': 9.974003647535408, 'STEPS_pareto': 0.6130675933667971, 'N_people_per_space': 27},
                        'office':           {'k': 2.8809814385623684, 'STEPS_pareto': 0.7676296683599062,'N_people_per_space': 14},
                        'supermarked':      {'k': 2.387294824952671, 'STEPS_pareto': 2.887134974372392, 'N_people_per_space': 11}
                        }
        if self.ntype in best_paras_STEPS:
            self.k = best_paras_STEPS[self.ntype]['k']
            self.STEPS_pareto = best_paras_STEPS[self.ntype]['STEPS_pareto']
            self.Npps = best_paras_STEPS[self.ntype]['N_people_per_space']
        else:
            raise ValueError(f"Unknown ntype: {self.ntype}")
        

        # Create DataFrame directly with the correct dtypes
        self.df_activities = pd.DataFrame({
            'loc_id_end': loc_id_end,
            'p_id': p_id,
            'activity_start_min': activity_start_min,
            'activity_end_min': activity_end_min
        })

        # Make location
        self.Loc  = build_location(self.df_activities, self.loc_id, self.ntype, N_pps=self.Npps, capacity=self.C)


    def make_network(self):
        CN = ContactNetwork(self.df_activities, self.Loc, self.t_start, self.t_end, time_scale_data=1)
        
        # Set optimized parameters adn simulate
        CN.k = self.k
        CN.STEPS_pareto = self.STEPS_pareto
        CN.make_movement('STEPS_pareto')
        CN.make_tacoma_network(min_dist=0, max_dist=1.5, time_resolution=20, export=False, temporal_offset=0)
        self.tn = CN.tn

    def aggregate_contact_network(self, TU_aggregated):
        self.TU_aggregated = TU_aggregated

        A = np.array(tc.adjacency_matrices(self.tn).adjacency_matrices)[:-1]

        T, N, N = A.shape
        # Ensure T is divisible by TU_aggregated
        assert T % TU_aggregated == 0, "T must be divisible by TU_aggregated"

        # Reshape the array to group T-axis in chunks of size TU_aggregated
        reshaped_arr = A.reshape(T // TU_aggregated, TU_aggregated, N, N)

        # Sum over the axis corresponding to the chunks
        A_agg = reshaped_arr.sum(axis=1) / TU_aggregated
        print(A_agg)

        np.save(f'./24h_networks/{self.loc_id}.npy', A_agg)
    






if __name__=='__main__':
    best_paras_STEPS =  {
                        # Dicitionary with STEPS paras after tuning, changed Npps to actual value
                        'highschool':       {'k': 4.3879605793130265, 'STEPS_pareto': 0.4210183226476539, 'N_people_per_space': 20},
                        'primaryschool':    {'k': 9.974003647535408, 'STEPS_pareto': 0.6130675933667971, 'N_people_per_space': 27},
                        'office':           {'k': 2.8809814385623684, 'STEPS_pareto': 0.7676296683599062,'N_people_per_space': 14},
                        'superamrked':      {'k': 2.387294824952671, 'STEPS_pareto': 2.887134974372392, 'N_people_per_space': 11}
                        }
    best_paras_ntype = {'highschool': {'STEPS_pareto': {'k': 4.3879605793130265,
                                'STEPS_pareto': 0.4210183226476539,
                                'N_people_per_space': 27},
                                'STEPS_with_RWP_pareto': {'k': 4.86415873950932,
                                'STEPS_pareto': 0.6189274019207999,
                                'N_people_per_space': 2,
                                'RWP_WT_MAX': 3324},
                                'clique_with_random': {'p_space_change': 0.0562632459155923,
                                'mean': 170,
                                'sigma': 91,
                                'p_add': 0.008016298535436495,
                                'pareto_shape': 9.998054452780455,
                                'N_people_per_space': 19},
                                'random': {'p_add': 0.00034618042207039445,
                                'pareto_shape': 7.061619041378239},
                                'baseline': {'weight': 0.0002992849250132354},
                                'RWP': {'RWP_WT_MAX': 3599},
                                'TLW': {'TLW_WT_MAX': 2359,
                                'TLW_WT_EXP': -0.5029329300743824,
                                'FL_MAX': 80,
                                'FL_EXP': -9.987833730779137}}, 
                        
                        'primaryschool': {'STEPS_pareto': {'k': 9.974003647535408,
                                'STEPS_pareto': 0.6130675933667971,
                                'N_people_per_space': 39},
                                'STEPS_with_RWP_pareto': {'k': 8.202328300072152,
                                'STEPS_pareto': 0.6998265484241938,
                                'N_people_per_space': 5,
                                'RWP_WT_MAX': 1133},
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
                                'FL_EXP': -8.996563363396556}}, 
                        
                        'office': {'STEPS_with_RWP_pareto': {'k': 4.372608807670108,
                            'STEPS_pareto': 0.7189126397421236,
                            'N_people_per_space': 30,
                            'RWP_WT_MAX': 1177},
                            'STEPS_pareto': {'k': 2.8809814385623684,
                            'STEPS_pareto': 0.7676296683599062,
                            'N_people_per_space': 24},
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
                            'FL_EXP': -9.894483858860687}},
                        
                        'supermarked': {'STEPS_with_RWP_pareto': {'k': 9.161041962098524,
                                'STEPS_pareto': 2.1718837262004,
                                'N_people_per_space': 24,
                                'RWP_WT_MAX': 24},
                                'clique_with_random': {'p_space_change': 0.00820944885032916,
                                'mean': 574,
                                'sigma': 43,
                                'p_add': 0.09139312807734347,
                                'pareto_shape': 2.1211790041196585,
                                'N_people_per_space': 31},
                                'random': {'p_add': 0.009703836159227257, 'pareto_shape': 1.4881299058280244},
                                'STEPS_pareto': {'k': 2.387294824952671,
                                'STEPS_pareto': 2.887134974372392,
                                'N_people_per_space': 20},
                                'TLW': {'TLW_WT_MAX': 2419,
                                'TLW_WT_EXP': -6.70681300692661,
                                'FL_MAX': 37,
                                'FL_EXP': -1.839834228938802},
                                'RWP': {'RWP_WT_MAX': 54},
                                'baseline': {'weight': 0.009970022805909147}}}
    
    '''MCN = MemilioContactNetwork('highschool', N=60)
    MCN.create_tapas_like_trajectories()
    MCN.make_network()
    MCN.aggregate_contact_network(180)'''

    df_base = pd.read_parquet('higschool_animation_traj.parquet')
    t_start, t_end = df_base.activity_start_min.min(), df_base.activity_end_min.max()


    Loc = build_location(df_base, 'hs_test', 'highschool', N_pps=23)
    CN = ContactNetwork(df_base, Loc, t_start, t_end, 20)
    CN.k = 1.42
    CN.STEPS_pareto = .3
    CN.RWP_WT_MAX = 20

    method = 'RWP'
    CN.make_movement(method=method)
    CN.animate_movement()
    
    # interpolation_test_singular(CN, round(1500), round(1800))
    # CN.make_tacoma_network(temporal_offset=1000)


    pass
