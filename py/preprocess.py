import os
from os.path import join
import logging
from typing import Tuple

import pandas as pd
import numpy as np
import nfl_data_py as nfl

from ball_path import estimate_ball_path

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
LOG = logging.getLogger(__name__)

DATA_DIR = '../data/'

def process_data(
    tracking_input: pd.DataFrame,
    tracking_output: pd.DataFrame,
    sup_data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process the raw data files into cleaned and standardized DataFrames.

    Args:
        tracking_input: The input tracking data.
        tracking_output: The output tracking data.
        sup_data: Supplemental data.    

    Returns:
        The games, plays, players and tracking DataFrames.
    """
    games, plays, players, tracking = join_split_standardize(
        tracking_input, tracking_output, sup_data
    )
    games, plays, tracking = add_nfl_pbp_info(games, plays, tracking)
    tracking, plays = estimate_ball_path(tracking, plays)

    return games, plays, players, tracking

def join_split_standardize(
    tracking_input: pd.DataFrame,
    tracking_output: pd.DataFrame,
    sup_data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Join the input and output tracking data. Split the data int plays, players and 
    tracking. Standardize the direction of the play and the players to be left to right, 
    where the offense is moving from the left to the right.

    NOTE: Imputing the missing speeds in the output data is very accurate (>.99 R²),
    but the missing accelerations are very inaccurate (~.05 R²). The missing
    directions are also fairly accurate (~.85 R²). The orientation is assumed
    to be equal to the direction when missing.

    Args:
        tracking_input: The input tracking data.
        tracking_output: The output tracking data.
        sup_data: Supplemental data.

    Returns:
        The games, plays, players and tracking data with the direction standardized.
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
        'player_to_predict', 'player_side', 'player_role', 'player_position', 'x', 'y', 
        's', 'a', 'dir', 'o']
    tracking = pd.concat([
        tracking_input[tracking_cols],
        tracking_output.merge(
            tracking_input[['game_id', 'play_id', 'nfl_id', 'player_to_predict', 
                            'player_side', 'player_role','player_position']].drop_duplicates(),
            on=['game_id', 'play_id', 'nfl_id'],
            how='left'
        ).assign(
            s=np.nan,
            a=np.nan,
            dir=np.nan,
            o=np.nan
        )
        [tracking_cols]
    ]).sort_values(['gpid', 'nfl_id', 'frame_id'], ignore_index=True).rename(columns={
        'player_position': 'position'
    })

    LOG.info('Joined input and output tracking data: %d unique plays, %d unique nfl_ids',
                n_unique_gpid, n_unique_nflid)

    # Standardize the direction of the play and the players to be left to right
    LOG.info('Standardizing direction of play and players to be left to right')
    tracking, plays = _standardize_direction(tracking, plays)

    # Approximate missing speed, acceleration and direction values for the output frames
    LOG.info('Approximating missing speed, acceleration and direction values')
    tracking = _approximate_missing_speed_acceleration_direction(tracking)

    # Assumption: missing orientation values equals direction values
    tracking['o'] = tracking['o'].fillna(tracking['dir'])

    LOG.info('Joining supplemental data to plays DataFrame')
    plays = join_supplemental_data(plays, sup_data)

    games = plays[['game_id','season','week','game_date','game_time_eastern',
        'home_team_abbr','visitor_team_abbr',]].drop_duplicates().reset_index(drop=True)
    
    plays.drop(columns=['game_date','game_time_eastern',
        'home_team_abbr','visitor_team_abbr'], inplace=True)

    return games, plays, players, tracking  

def _standardize_direction(
    tracking: pd.DataFrame,
    play: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize the direction of the play and the players to be left to right,

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
    
    # Standardize the player data to be left to right
    original_x = tracking['x'].copy()
    original_y = tracking['y'].copy()
    original_dir = tracking['dir'].copy()
    original_o = tracking['o'].copy()
    tracking['x'] = np.where(
        left_play, 
        120 - original_x,
        original_x
    )
    tracking['y'] = np.where(
        left_play, 
        53.3 - original_y,
        original_y
    )
    tracking['dir'] = np.where(
        original_dir.isna(),
        np.nan,
        np.where(
            left_play,
            (original_dir + 180) % 360,
            original_dir
        )
    )
    tracking['o'] = np.where(
        original_o.isna(),
        np.nan,
        np.where(
            left_play,
            (original_o + 180) % 360,
            original_o
        )
    )
    tracking = tracking.drop('play_direction', axis=1)

    # Standardize the play data to be left to right
    play['absolute_yardline_number'] = np.where(
        play.play_direction == "left", 
        120 - play.absolute_yardline_number, 
        play.absolute_yardline_number
    )
    original_ball_land_x = play['ball_land_x'].copy()
    original_ball_land_y = play['ball_land_y'].copy()
    play['ball_land_x'] = np.where(
        play.play_direction == "left", 
        120 - original_ball_land_x,
        original_ball_land_x
    )
    play['ball_land_y'] = np.where(
        play.play_direction == "left", 
        53.3 - original_ball_land_y,
        original_ball_land_y
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
            correlations = (f'{s_col}: speed R²={s_r2:.4f} | {a_col}: accel R²={a_r2:.4f}'
                f' | {dir_col}: dir R²={dir_r2:.4f}')
            LOG.info('Correlation results for imputations: %s', correlations)

    # Fill in missing values with the approximations
    tracking['s'] = tracking['s'].fillna(tracking['s_approx'])
    tracking['a'] = tracking['a'].fillna(tracking['a_approx'])
    tracking['dir'] = tracking['dir'].fillna(tracking['dir_approx'])
    tracking = tracking.drop(columns=['x_prev', 'y_prev', 'dt', 
                                      's_approx', 'a_approx', 'dir_approx'])
    return tracking

def join_supplemental_data(
    plays: pd.DataFrame,
    sup_data: pd.DataFrame
) -> pd.DataFrame:
    """Join supplemental data to the plays DataFrame.

    Args:
        plays: The plays DataFrame.
        sup_data: The supplemental data DataFrame.

    Returns:
        The plays DataFrame with the supplemental data joined.
    """

    sup_data['gpid'] = sup_data['game_id'].astype(str) + '_' + \
        sup_data['play_id'].astype(str)
    
    sup_data_keep_cols = ['gpid', 'season', 'week', 'game_date', 'game_time_eastern',
       'home_team_abbr', 'visitor_team_abbr', 'play_description',
       'quarter', 'game_clock', 'down', 'yards_to_go', 'possession_team',
       'defensive_team', 'yardline_side', 'yardline_number',
       'pre_snap_home_score', 'pre_snap_visitor_score',
       'play_nullified_by_penalty', 'pass_result', 'pass_length',
       'offense_formation', 'receiver_alignment', 'route_of_targeted_receiver',
       'play_action', 'dropback_type', 'dropback_distance',
       'pass_location_type', 'defenders_in_the_box', 'team_coverage_man_zone',
       'team_coverage_type', 'penalty_yards', 'pre_penalty_yards_gained',
       'yards_gained', 'expected_points', 'expected_points_added',
       'pre_snap_home_team_win_probability',
       'pre_snap_visitor_team_win_probability',
       'home_team_win_probability_added', 'visitor_team_win_probility_added']
    plays = plays.merge(
        sup_data[sup_data_keep_cols],
        on='gpid',
        how='left'
    )
    return plays


['old_game_id','play','passer_player_id','receiver_player_id','interception_player_id',
 'yards_gained','air_yards','yards_after_catch','weather','roof','temp','wind']

def add_nfl_pbp_info(
    games: pd.DataFrame,
    plays: pd.DataFrame,
    tracking: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add NFL PBP information to the plays and tracking DataFrames.

    Args:
        games: The games DataFrame.
        plays: The plays DataFrame.
        tracking: The tracking DataFrame.
    Returns:
        The games, plays and tracking DataFrames with the NFL PBP information added.
    """
    seasons = plays.season.unique().tolist()

    # Load NFL PBP data
    pbps = []
    if not os.path.exists(os.path.join(DATA_DIR, 'pbp')):
        os.makedirs(os.path.join(DATA_DIR, 'pbp'))
    for season in seasons:
        LOG.info('Loading NFL PBP data for season %d', season)
        # check if pbp for that season is already downloaded at DATA_DIR pbp/season.parquet
        file_path = os.path.join(DATA_DIR, 'pbp', f'{season}.parquet')
        if os.path.exists(file_path):
            LOG.info('Loading pbp from local parquet file')
            pbp = pd.read_parquet(file_path)
            pbps.append(pbp)
        else:
            # Using direct URL as the import_pbp_data function seems to be having issues
            url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
            try:
                pbp = pd.read_parquet(url)
                LOG.info("Successfully loaded from direct URL")
                pbps.append(pbp)
                LOG.info('Saving pbp to local parquet file')
                pbp.to_parquet(file_path, index=False)
            except Exception as e:
                LOG.error(f"URL method failed: {e}")
    pbp = pd.concat(pbps, ignore_index=True)
    pbp['gpid'] = pbp['old_game_id'].astype(str) + '_' + pbp['play_id'].astype(int).astype(str)

    # Map player IDs to nfl_id using seasonal rosters
    if not os.path.exists(os.path.join(DATA_DIR, 'rosters')):
        os.makedirs(os.path.join(DATA_DIR, 'rosters'))
    LOG.info('Mapping player IDs to nfl_id using seasonal rosters')
    rosters_list = []
    for season in seasons:
        file_path = os.path.join(DATA_DIR, 'rosters', f'{season}.parquet')
        if not os.path.exists(file_path):
            LOG.info('Caching rosters for season %d', season)
            rosters = nfl.import_seasonal_rosters(years=[season])
            rosters.to_parquet(file_path, index=False)
        else:
            LOG.info('Rosters for season %d already cached, loading from parquet', season)
            rosters = pd.read_parquet(file_path)
        rosters_list.append(rosters)
    rosters = pd.concat(rosters_list, ignore_index=True)
    player_id_to_nfl_id = (
        rosters[['gsis_it_id','player_id']]
        .drop_duplicates()
        .rename(columns={'gsis_it_id': 'nfl_id'})
    ).to_dict(orient='records')
    for col in ['passer_player_id','receiver_player_id','interception_player_id']:
        pbp[col] = pbp[col].map(lambda x: 
            next((item['nfl_id'] for item in player_id_to_nfl_id if item['player_id'] == x), np.nan)
        )

    # Merge relevant PBP columns to plays
    play_pbp_cols = ['gpid', 'yards_gained','air_yards','yards_after_catch']
    plays = plays.merge(
        pbp[play_pbp_cols],
        on='gpid',
        how='left'
    )

    # Merge relevant PBP columns to games
    game_pbp_cols = ['old_game_id','weather','roof','temp','wind']
    games = games.merge(
        pbp[game_pbp_cols].drop_duplicates(['old_game_id'])
            .rename(columns={'old_game_id': 'game_id'})
                .assign(game_id=lambda df: df['game_id'].astype(int)),
        on='game_id',
        how='left'
    )

    # Set the passer, receiver, and interceptor flags in the tracking DataFrame
    tracking_pbp_cols = ['gpid', 'passer_player_id','receiver_player_id','interception_player_id']
    tracking = tracking.merge(
        pbp[tracking_pbp_cols],
        on='gpid',
        how='left'
    )
    tracking = tracking.assign(
        passer_player_id = tracking['passer_player_id'].fillna(0).astype(int),
        receiver_player_id = tracking['receiver_player_id'].fillna(0).astype(int),
        interception_player_id = tracking['interception_player_id'].fillna(0).astype(int)
    )
    tracking = tracking.assign(
        is_passer = tracking['nfl_id'] == tracking['passer_player_id'],
        is_receiver = tracking['nfl_id'] == tracking['receiver_player_id'],
        is_interceptor = tracking['nfl_id'] == tracking['interception_player_id']
    ).drop(columns=['passer_player_id','receiver_player_id','interception_player_id'])

    # if tracking gpid does not have a passer, set it to the qb
    gpid_without_passer = set(tracking.gpid.unique()) - \
        set(tracking.query('is_passer').gpid.unique())
    if len(gpid_without_passer) > 0:
        for gpid in gpid_without_passer:
            if tracking.query('gpid == @gpid and position == "QB"').empty:
                LOG.warning('Dropping gpid %s with no passer and no QB', gpid)
                tracking = tracking[tracking.gpid != gpid]
                plays = plays[plays.gpid != gpid]
            else:
                LOG.info('Defaulting passer to QB for play without a passer: %s', gpid)
                qb = (
                    tracking.query('position == "QB" and gpid == @gpid')
                    [['gpid','nfl_id']].drop_duplicates()
                )
                qb = qb.rename(columns={'nfl_id': 'passer_nfl_id'})
                tracking = tracking.merge(qb, on='gpid', how='left')
                tracking['is_passer'] = np.where(
                    tracking.gpid.isin(gpid_without_passer),
                    tracking['nfl_id'] == tracking['passer_nfl_id'],
                    tracking['is_passer']
                )
                tracking = tracking.drop(columns=['passer_nfl_id'])

    return games, plays, tracking

def fetch_team_desc() -> pd.DataFrame():
    team_desc_file = os.path.join(DATA_DIR, 'team_desc.parquet')
    if os.path.exists(team_desc_file):
        LOG.info('Loading team description info from cached parquet.')
        team_desc = pd.read_parquet(team_desc_file)
    else:
        LOG.info('Loading team description info from nfl-data-py.')
        team_desc = nfl.import_team_desc()
        team_desc.to_parquet(team_desc_file, index=False)
    return team_desc