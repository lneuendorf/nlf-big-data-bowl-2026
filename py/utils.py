import re
from typing import Tuple

import pandas as pd
import numpy as np

def uncamelcase_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert camelCase column names to snake_case."""
    df.columns = [re.sub(r'(?<!^)(?=[A-Z])', '_', word).lower() for word in df.columns]
    return df

def join_split_standardize(
    tracking_input: pd.DataFrame,
    tracking_output: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Join the input and output tracking data. Split the data int plays, players and 
    tracking. Standardize the direction of the play and the players to be vertical, 
    where the offense is moving from the bottom to the top of the field.

    NOTE: Imputing the missing speeds in the output data is very accurate (>.99 R²),
    but the missing accelerations are very inaccurate (~.05 R²). The missing
    directions are also fairly accurate (~.85 R²). The orientation is assumed
    to be equal to the direction when missing.

    Args:
        tracking_input: The input tracking data.
        tracking_output: The output tracking data.

    Returns:
        The plays, players and tracking data with the direction standardized.
    """

    # Create a unique identifier for each game + play combination
    tracking_input['gpid'] = tracking_input['game_id'].astype(str) + '_' + \
        tracking_input['play_id'].astype(str)
    tracking_output['gpid'] = tracking_output['game_id'].astype(str) + '_' + \
        tracking_output['play_id'].astype(str)
    n_unique_gpid = tracking_input.gpid.nunique()
    n_unique_nflid = tracking_input.nfl_id.nunique()

    # Boolean pass thrown indicator
    tracking_input['pass_thrown'] = False
    tracking_output['pass_thrown'] = True

    # Adjust the output frame ids to be continuous with the input frame ids
    max_input_frameid = tracking_input.groupby('gpid')['frame_id'].max().reset_index()
    max_input_frameid = max_input_frameid.rename(columns={'frame_id': 'max_input_frame_id'})
    tracking_output = tracking_output.merge(max_input_frameid, on='gpid', how='left')
    tracking_output['frame_id'] = tracking_output['frame_id'] + tracking_output['max_input_frame_id']
    tracking_output = tracking_output.drop(columns=['max_input_frame_id'])

    # Play level data
    play_cols = ['gpid', 'game_id', 'play_id', 'play_direction', 
        'absolute_yardline_number', 'num_frames_output', 'ball_land_x', 'ball_land_y']
    if n_unique_gpid != tracking_input[play_cols].drop_duplicates().shape[0]:
        raise ValueError('Play data different across rows for same gpid')
    plays = tracking_input[play_cols].drop_duplicates(['gpid'], ignore_index=True)

    # Player level data
    player_cols = ['nfl_id', 'player_name', 'player_height', 'player_weight', 
        'player_birth_date', 'player_position']
    if n_unique_nflid != tracking_input[player_cols].drop_duplicates().shape[0]:
        raise ValueError('Player data different across rows for same nfl_id')
    players = tracking_input[player_cols].drop_duplicates('nfl_id', ignore_index=True)

    # Tracking level data by frame
    tracking_cols = ['gpid', 'game_id', 'play_id', 'frame_id', 'nfl_id', 'pass_thrown',
        'player_to_predict', 'player_side', 'player_role', 'x', 'y', 's', 'a', 'dir', 'o']
    tracking = pd.concat([
        tracking_input[tracking_cols],
        tracking_output.merge(
            tracking_input[['game_id', 'play_id', 'nfl_id', 'player_to_predict', 
                            'player_side', 'player_role']].drop_duplicates(),
            on=['game_id', 'play_id', 'nfl_id'],
            how='left'
        ).assign(
            s=np.nan,
            a=np.nan,
            dir=np.nan,
            o=np.nan
        )
        [tracking_cols]
    ]).sort_values(['gpid', 'nfl_id', 'frame_id'], ignore_index=True)

    # Standardize the direction of the play and the players to be vertical
    tracking, plays = _standardize_direction(tracking, plays)

    # Approximate missing speed, acceleration and direction values for the output frames
    tracking = _approximate_missing_speed_acceleration_direction(tracking)

    # Assumption: missing orientation values equals direction values
    tracking['o'] = tracking['o'].fillna(tracking['dir'])

    return plays, players, tracking    

def _standardize_direction(
    tracking: pd.DataFrame,
    play: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize the direction of the play and the players to be vertical.

    The direction of the play is set to be bottom to top, with the offensive
    moving from the bottom to the top.

    Args:
        tracking: The tracking data.
        play: The play data.

    Returns:
        The tracking data and the play data with the direction standardized.
    """
    tracking = tracking.merge(
        play[['game_id','play_id','play_direction']].drop_duplicates(['game_id','play_id']),
        on=['game_id','play_id'],
        how='left'
    )
    left_play = tracking['play_direction'] == 'left'
    
    # Standardize the player data to be vertical
    original_x = tracking['x'].copy()
    original_y = tracking['y'].copy()
    original_dir = tracking['dir'].copy()
    original_o = tracking['o'].copy()
    tracking['x'] = np.where(
        left_play, 
        original_y,
        53.3 - original_y
    )
    tracking['y'] = np.where(
        left_play, 
        120 - original_x,
        original_x
    )
    tracking['dir'] = np.where(
        original_dir.isna(),
        np.nan,
        np.where(
            left_play,
            (((180 - original_dir) % 360) + 180) % 360,
            (180 - original_dir) % 360
        )
    )
    tracking['o'] = np.where(
        original_o.isna(),
        np.nan,
        np.where(
            left_play,
            (((180 - original_o) % 360) + 180) % 360,
            (180 - original_o) % 360
        )
    )

    # Standardize the play data to be vertical
    play['absolute_yardline_number'] = np.where(
        play.play_direction == "left", 
        120 - play.absolute_yardline_number, 
        play.absolute_yardline_number
    )
    original_ball_land_x = play['ball_land_x'].copy()
    original_ball_land_y = play['ball_land_y'].copy()
    play['ball_land_x'] = np.where(
        play.play_direction == "left", 
        original_ball_land_y,
        53.3 - original_ball_land_y
    )
    play['ball_land_y'] = np.where(
        play.play_direction == "left", 
        120 - original_ball_land_x,
        original_ball_land_x
    )
    play = play.drop('play_direction', axis=1)

    return tracking, play

def _approximate_missing_speed_acceleration_direction(
    tracking: pd.DataFrame
) -> pd.DataFrame:
    """Approximate missing speed, acceleration, and direction values with multiple methods."""

    tracking = tracking.sort_values(['gpid', 'nfl_id', 'frame_id'], ignore_index=True)

    tracking[['x_prev', 'y_prev']] = tracking.groupby(['gpid', 'nfl_id'])[['x', 'y']].shift(1)
    tracking = tracking.assign(dt=0.1)
    
    # --- 2-point method (prev and current point) ---
    tracking['s_approx'] = np.round(
        np.sqrt((tracking['x'] - tracking['x_prev'])**2 + 
                (tracking['y'] - tracking['y_prev'])**2) / tracking['dt'],
        2)
    tracking['a_approx'] = np.round(
        (tracking['s_approx'] - tracking.groupby(['gpid', 'nfl_id'])['s_approx'].shift(1)) / 
        tracking['dt'], 2)
    tracking['dir_approx'] = np.round(
        np.degrees(np.arctan2(tracking['y'] - tracking['y_prev'], 
        tracking['x'] - tracking['x_prev'])) % 360, 2)

    # --- Correlation tests ---
    methods = [
        ('s_approx', 'a_approx', 'dir_approx'),
    ]
    for s_col, a_col, dir_col in methods:
        nonull = tracking.query(f's.notna() and a.notna() and {s_col}.notna() and {a_col}.notna()')
        if len(nonull) > 100:
            s_r2 = nonull[['s', s_col]].corr().iloc[0,1]**2
            a_r2 = nonull[['a', a_col]].corr().iloc[0,1]**2
            dir_r2 = nonull[['dir', dir_col]].corr().iloc[0,1]**2
            print(f'{s_col}: speed R²={s_r2:.4f} | {a_col}: accel R²={a_r2:.4f}'
                  f' | {dir_col}: dir R²={dir_r2:.4f}')

    # Fill in missing values with the approximations
    tracking['s'] = tracking['s'].fillna(tracking['s_approx'])
    tracking['a'] = tracking['a'].fillna(tracking['a_approx'])
    tracking['dir'] = tracking['dir'].fillna(tracking['dir_approx'])
    tracking = tracking.drop(columns=['x_prev', 'y_prev', 'dt', 
                                      's_approx', 'a_approx', 'dir_approx'])
    return tracking