import logging

import pandas as pd
import numpy as np

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
LOG = logging.getLogger(__name__)

def estimate_ball_path(tracking: pd.DataFrame, plays: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate ball path and assign constant speed and direction for in-air frames.
    """
    LOG.info("Estimating ball path for each play")

    # --- Filter to plays with exactly one passer and receiver ---
    valid_gpids = []
    for gpid, group in tracking.drop_duplicates(['gpid', 'nfl_id']).groupby('gpid'):
        num_passers = group['is_passer'].sum()
        num_receivers = group['is_receiver'].sum()
        if num_passers == 1 and num_receivers == 1:
            valid_gpids.append(gpid)
        else:
            LOG.warning(
                "Dropping gpid %s with %d passers and %d receivers",
                gpid, num_passers, num_receivers,
            )

    tracking = tracking[tracking['gpid'].isin(valid_gpids)].copy()
    plays = plays[plays['gpid'].isin(valid_gpids)].copy()

    # --- Set the ball location to passer location before the pass ---
    ball_paths = (
        tracking.query('is_passer')
        .reset_index(drop=True)
        .assign(
            nfl_id=0,
            player_to_predict=False,
            player_side='Ball',
            player_role='Ball',
            position='Ball',
            x=lambda df: np.where(df.pass_thrown, np.nan, df.x),
            y=lambda df: np.where(df.pass_thrown, np.nan, df.y),
            s=lambda df: np.where(df.pass_thrown, np.nan, df.s),
            a=np.nan,
            dir=lambda df: np.where(df.pass_thrown, np.nan, df.dir),
            o=np.nan,
            is_passer=False,
            is_receiver=False,
            is_interceptor=False,
        )
    )

    frame_time = 0.1  # seconds between frames
    pass_dists = {}
    air_dfs = []

    # --- Estimate ball path in air ---
    for gpid, group in ball_paths.groupby('gpid', sort=False):
        group = group.sort_values('frame_id').copy()
        play_meta = plays.loc[plays['gpid'] == gpid]
        if play_meta.empty:
            continue

        # Destination info
        ball_land_x = play_meta['ball_land_x'].values[0]
        ball_land_y = play_meta['ball_land_y'].values[0]
        num_frames_air = int(play_meta['num_frames_output'].values[0])

        first_air_frame = group.frame_id.max() + 1

        # Origin of pass (when ball leaves QB)
        passer_prev = group[group['frame_id'] == first_air_frame - 1]
        if passer_prev.empty:
            passer_prev = group[group['frame_id'] == first_air_frame]
        start_x, start_y = passer_prev.iloc[0][['x', 'y']]

        # End coordinates
        end_x, end_y = ball_land_x, ball_land_y

        # --- Compute distance and constant velocity ---
        dx, dy = end_x - start_x, end_y - start_y
        dist_xy = np.sqrt(dx**2 + dy**2)
        pass_dists[gpid] = dist_xy

        total_time = num_frames_air * frame_time
        ball_speed = dist_xy / total_time if total_time > 0 else 0.0
        ball_dir = round(np.degrees(np.arctan2(dy, dx)) % 360, 2)

        # --- Linear interpolation for positions ---
        x_path = np.linspace(start_x, end_x, num_frames_air + 1)
        y_path = np.linspace(start_y, end_y, num_frames_air + 1)

        in_air_frames = range(first_air_frame, first_air_frame + num_frames_air)

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
            'x': x_path[1:],  # exclude QB hand point
            'y': y_path[1:],
            's': ball_speed,  # constant horizontal speed
            'a': 0.0,
            'dir': ball_dir,  # constant direction (radians)
            'o': np.nan,
            'is_passer': False,
            'is_receiver': False,
            'is_interceptor': False,
        })

        air_dfs.append(air_df)

    # --- Combine all results ---
    tracking = pd.concat([tracking, ball_paths, *air_dfs], ignore_index=True)
    tracking = tracking.sort_values(['gpid', 'nfl_id', 'frame_id'], ignore_index=True)

    plays['pass_distance'] = plays['gpid'].map(pass_dists)

    LOG.info("Finished estimating ball path for %d plays", len(plays))
    return tracking, plays
