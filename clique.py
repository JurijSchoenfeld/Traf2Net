from util import boolean_blocks_indices, map_pid
import numpy as np
import pandas as pd


def get_contacts(location):
    start_times, end_times = location.activity_start_min.values, location.activity_end_min.values
    event_ids = location.p_id.values
    loc_id_end = location.loc_id_end.values[0]
    # Takes a df containing all activities at location
    # Returns all possible contacts

    # Broadcast the start_time and end_time arrays for comparison with each other
    overlap_start = np.maximum.outer(start_times, start_times)
    overlap_end = np.minimum.outer(end_times, end_times)

    # Calculate the overlap duration matrix 
    overlap_durations = np.maximum(overlap_end - overlap_start, np.zeros(shape=overlap_start.shape)).astype('uint16')
    
    # Set diagonal elements to zero (overlap of an event with itself and double counting)
    overlap_durations = np.triu(overlap_durations, 1)

    # Extract contact rows, cols
    rows, cols = np.where(overlap_durations > 0)
    p_A = event_ids[rows].astype('int')

    # Save contacts to new DataFrame
    contact_data = {'p_A': p_A,'p_B': event_ids[cols].astype('int'), 
                    'start_of_contact': overlap_start[rows, cols].astype('int'),
                    'end_of_contact': overlap_end[rows, cols].astype('int'),
                    'loc_id': np.repeat(loc_id_end, len(p_A)).astype('int32')}

    return pd.DataFrame(contact_data)


def assign_spaces(location, N_peaoplePerSpace, p_space_change, mean, sigma):
    np.random.seed(1)
    # Get number of spaces
    nodes = location.p_id.unique()
    N_spaces = int(np.ceil(nodes.size / N_peaoplePerSpace)) + 1

    # Boundaries for time series
    ts0, tsmax = location.activity_start_min.min(), location.activity_end_min.max()
    time_series = np.zeros(shape=(nodes.size, tsmax - ts0))

    # Get default locations
    default_spaces = dict(zip(nodes, np.random.randint(1, N_spaces, size=nodes.size)))

    # Group people
    p_grouped = location.groupby('p_id')


    def match_time(trajectory, p_id, default_space):
        # this method return a time series that contains the space of each node, np.nan means the node is not present
        # Go trough trajectories
        # Assign default space
        trajectory_time_series = np.zeros(shape=(tsmax - ts0))  # zero corresponds to node is not in location
        tr_start, tr_end = trajectory.activity_start_min, trajectory.activity_end_min
        trajectory_time_series[tr_start: tr_end] = default_space

        # Assign random space
        space_changes = (np.arange(tr_start, tr_end, 1)[np.random.rand(tr_end - tr_start) < p_space_change]).astype('int')
        spaces = np.random.randint(1, N_spaces, space_changes.size).astype('int')
        space_change_durations = np.rint(np.random.normal(mean, sigma, space_changes.size)).astype('int')
        space_change_durations[space_change_durations < 1] = 1

        for space_change, space, duration in zip(space_changes, spaces, space_change_durations):
            trajectory_time_series[space_change: space_change + duration] = space
        
        time_series[p_id][trajectory_time_series > 0] = trajectory_time_series[trajectory_time_series > 0]

    def get_time_series(p_group):
        # Go through people
        default_space = default_spaces[p_group.name]
        return p_group.apply(match_time, axis=1, args=(p_group.name, default_space))

    p_grouped.apply(get_time_series)
    time_series[time_series == 0] = np.nan
    return time_series


def check_space(contact, time_series, p_As, p_Bs, start_contacs, end_contacts, spaces):
    # Checks wether two nodes simultaniously present at location are at the same space
    p_A, p_B = contact.p_A, contact.p_B

    space_overlap = time_series[p_A, contact.start_of_contact: contact.end_of_contact] == time_series[p_B, contact.start_of_contact: contact.end_of_contact]
    space_contact_start, space_contact_end = boolean_blocks_indices(space_overlap)
    space_contact_start += contact.start_of_contact
    space_contact_end += contact.start_of_contact

    p_As.extend(np.repeat(p_A, space_contact_start.size))
    p_Bs.extend(np.repeat(p_B, space_contact_start.size))
    start_contacs.extend(space_contact_start)
    end_contacts.extend(space_contact_end)
    spaces.extend(time_series[p_A][space_contact_start])


def get_contacts_spaces(location, N_peaoplePerSpace, p_space_change, mean, sigma):
    # Map ids
    location = map_pid(location)

    contacts = get_contacts(location)
    time_series = assign_spaces(location, N_peaoplePerSpace, p_space_change, mean, sigma)

    p_As, p_Bs, start_contacs, end_contacts, spaces = [], [], [], [], []
    contacts.apply(check_space, axis=1, args=(time_series, p_As, p_Bs, start_contacs, end_contacts, spaces))

    space_contacts = pd.DataFrame({'p_A': p_As, 'p_B': p_Bs, 'start_of_contact': start_contacs, 'end_of_contact': end_contacts, 'space': spaces})

    space_contacts = space_contacts[(space_contacts.end_of_contact - space_contacts.start_of_contact) > 0]

    return space_contacts
