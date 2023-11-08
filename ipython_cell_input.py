import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Geod
from tqdm.notebook import tqdm
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

df = pd.read_csv('./VF_data/pandemos_tra_tapas_modell.csv')

def get_nnodes(df, n):
    # Returns a list of n lists where each list contains all indices in the df of a certain node
    # The n selected nodes are choosen in order of appearence in df
    # Expects the df to be sorted by p_id and start_time_min

    unique_nodes = df.p_id.unique()
    nnodes = []
    
    if len(unique_nodes) < n: # select at maximum all available nodes in df
        n = len(unique_nodes)

    last_ind, first_ind = 0, 0
    for node in unique_nodes[:n]:
        while True:
            if df.iloc[first_ind].p_id != node:
                nnodes.append(list(range(last_ind, first_ind)))
                break

            first_ind += 1
        
        last_ind = first_ind
        
    
    return nnodes, unique_nodes

def get_position_vector_nnodes(df, n):
    index, unique_nnodes = get_nnodes(df, n)
    geod = Geod("+ellps=WGS84")
    R_lons, R_lats = [], []

    # get edges of time interval
    index_flat = [item for sublist in index for item in sublist]
    tmin = df.iloc[index_flat].start_time_min.min()

    if tmin < 0:
        tmin = abs(tmin)
    else:
        tmin = 0
    tmax = (df.iloc[index_flat].activity_start_min + df.iloc[index_flat].activity_duration_min).max() + tmin

    R_lons, R_lats = np.empty(tmax), np.empty(tmax)

    for i, (ind, id) in tqdm(enumerate(zip(index, unique_nnodes)), total=n):
        # get all df entries of current node
        dfj = df.iloc[ind].sort_values('start_time_min')

        # convert trajectories and activites into position vector R(t)=((lon_0, lat_1), (lon_1, lat_1), ..., (lon_max, lat_max))^T
        S_loni, S_lati = dfj.lon_start.to_numpy(), dfj.lat_start.to_numpy()
        F_loni, F_lati = dfj.lon_end.to_numpy(), dfj.lat_end.to_numpy()
        T = (dfj.travel_time_sec / 60).round(0).astype(int).to_numpy()
        At = dfj.activity_duration_min.to_numpy()
        wait = dfj.start_time_min.iloc[0] + tmin

        R = [(np.nan, np.nan),]*wait
        for lon1, lat1, lon2, lat2, npts, Ati in zip(S_loni, S_lati, F_loni, F_lati, T, At):
            Gi = geod.npts(lon1, lat1, lon2, lat2, npts, initial_idx=0, terminus_idx=0)
            Ai = [(lon2, lat2),]*Ati
            R += Gi + Ai

        R += [(np.nan, np.nan),]*(tmax - len(R))
        R = np.array(R).T
        
        try:
            R_lons = np.vstack((R_lons, R[0]))
            R_lats = np.vstack((R_lats, R[1]))
        except ValueError:
            over_shoot = len(R[0]) - tmax
            print(f'Over shoot due to precission error: {over_shoot}min')
            R_lons = np.vstack((R_lons, R[0, over_shoot:]))
            R_lats = np.vstack((R_lats, R[1, over_shoot:]))

    # Return in format R_x(t)=[[r1(t=0), r2(t=0), ..., rn(t=0)], [r1(t=1), r2(t=1), ..., rn(t=tmax)], ..., [r1(t=tmax), r2(t=tmax), ..., rn(t=tmax)]]
    return R_lons.T, R_lats.T
        
R_lons, R_lats = get_position_vector_nnodes(df, 10_000)
