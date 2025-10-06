import logging

import pandas as pd
import numpy as np

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
LOG = logging.getLogger(__name__)

def estimate_ball_path(
    tracking: pd.DataFrame,
    plays: pd.DataFrame
) -> pd.DataFrame:
    
    LOG.info('Estimating ball path for each play')

    # --- Filter to plays with exactly one passer and receiver ---
    valid_gpids = []
    for gpid, group in tracking.drop_duplicates(['gpid','nfl_id']).groupby('gpid'):
        num_passers = group['is_passer'].sum()
        num_receivers = group['is_receiver'].sum()
        if num_passers == 1 and num_receivers == 1:
            valid_gpids.append(gpid)
        else:
            LOG.warning('Dropping gpid %s with %d passers and %d receivers', 
                     gpid, num_passers, num_receivers)

    tracking = tracking[tracking['gpid'].isin(valid_gpids)].copy()
    plays = plays[plays['gpid'].isin(valid_gpids)].copy()

    # Constants
    frame_time = 0.1  # seconds between frames
    g = 9.81  # gravity (m/s^2)
    yard_to_m = 0.9144  # convert yards to meters

    ball_paths = (
        tracking.copy()
        .query('is_passer')
        .reset_index(drop=True)
        .assign(
            nfl_id=0,
            player_to_predict=False,
            player_side='Ball',
            player_role='Ball',
            position='Ball',
            x=lambda x: np.where(
                x.pass_thrown, np.nan, x.x
            ),
            y=lambda x: np.where(
                x.pass_thrown, np.nan, x.y
            ),
            s=np.nan,
            a=np.nan,
            dir=np.nan,
            o=np.nan,
            is_passer=False,
            is_receiver=False,
            is_interceptor=False,
        )
    )

    pass_dists = {}
    air_dfs = []
    for gpid, group in ball_paths.groupby('gpid', sort=False):
        group = group.sort_values('frame_id').copy()

        # Pull play info
        play_meta = plays.loc[plays['gpid'] == gpid]
        if play_meta.empty:
            continue
        ball_land_x = play_meta['ball_land_x'].values[0]
        ball_land_y = play_meta['ball_land_y'].values[0]
        num_frames_air = play_meta['num_frames_output'].values[0]

        # Find throw start frame (first frame with pass_thrown == True)
        first_air_frame = group.frame_id.max() + 1

        # Coordinates where pass leaves QB hand
        passer_prev = group[(group['frame_id'] == first_air_frame - 1)]
        if passer_prev.empty:
            passer_prev = group[group['frame_id'] == first_air_frame]
        start_x, start_y = passer_prev.iloc[0][['x', 'y']]

        # End coordinates from plays
        end_x, end_y = ball_land_x, ball_land_y

        # --- Compute projectile-like XY trajectory ---
        # total flight time
        t_total = round((num_frames_air + 1) * frame_time, 1)
        t_vals = np.linspace(0, t_total, num_frames_air + 1)

        # horizontal distance in yards
        dx = end_x - start_x
        dy = end_y - start_y
        dist_xy = np.sqrt(dx**2 + dy**2)
        pass_dists[gpid] = dist_xy

        # --- Solve for launch angle θ (in radians) ---
        # θ = arctan((g * t^2) / (2 * R))
        theta = np.arctan((g * (t_total ** 2)) / (2 * (dist_xy * yard_to_m)))

        # Compute initial velocity v0 in m/s, using vertical motion constraint
        v0 = (g * t_total) / (2 * np.sin(theta))
        v0_yards = v0 / yard_to_m  # convert to yards/s

        # --- Parametric x–y assuming equal start/end height ---
        x_path = start_x + dx * (t_vals / t_total)
        # Add realistic parabolic height bump based on vertical component
        arc_height = (v0_yards * np.sin(theta) * (t_total / 2)) / 10  # scale down for visible arc
        y_path = start_y + dy * (t_vals / t_total) + (-4 * (t_vals - t_total/2)**2 / t_total**2 + 1) * arc_height

        # --- Assign ball positions ---
        in_air_frames = range(first_air_frame, first_air_frame + num_frames_air)
        
        # append the in air frames to the group dataframe with all other columns duplicated as above
        air_df = pd.DataFrame({
            'gpid': gpid,
            'game_id': group['game_id'].values[0],
            'play_id': group['play_id'].values[0],
            'frame_id': in_air_frames,
            'nfl_id': 0,
            'pass_thrown': True,
            'player_to_predict': False,
            'player_side': 'Ball',
            'player_role': 'Ball',
            'position': 'Ball',
            'x': x_path[1:],  # exclude the first point which is the passer
            'y': y_path[1:],  # exclude the first point which is the passer
            's': np.nan,   
            'a': np.nan,
            'dir': np.nan,
            'o': np.nan,
            'is_passer': False,
            'is_receiver': False,
            'is_interceptor': False
        })

        air_dfs.append(air_df)

    tracking = pd.concat([tracking, ball_paths, *air_dfs], ignore_index=True)
    tracking = tracking.sort_values(['gpid', 'nfl_id', 'frame_id'], ignore_index=True)

    plays['pass_distance'] = plays['gpid'].map(pass_dists)

    return tracking, plays