"""
Filter:
- drop "throw away" plays where ball badely misses targeted receiver? These add noise to epa prediction, as if it was a throw away the defense always "wins"
- focus specifically on free safety starting at least 8 yards downfield
- pass must be at least x yards in air (safety needs time to react)

Then find set of points safety can reach: pass into epa model plus actual point
Find min epa of points vs actual point
Save to csv with relevant info
"""
import logging
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch

from preprocess import preprocess
from models.defender_reach.preprocessor import DefenderReachDataset
from models.defender_reach.model import DefenderReachModel
from models.epa.graph_dataset import EPAGraphDataset
from models.epa.epa_gnn_model import train_model
from models.safety_reachable_points.safety_reach import (
    simulate_outer_points,
    fill_polygon_with_grid
)

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RANDOM_SEED = 2
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
N_WEEKS = 18

# Was getting segfaults without these settings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

##############  i. Load and preprocess the data ##############
sup_data = pd.read_csv('../data/supplementary_data.csv')
tracking_input, tracking_output = pd.DataFrame(), pd.DataFrame()
for week in tqdm(range(1, N_WEEKS+1), desc="Loading weekly data"):
    tracking_input = pd.concat([tracking_input, pd.read_csv(f'../data/train/input_2023_w{week:02d}.csv')], axis=0)
    tracking_output = pd.concat([tracking_output, pd.read_csv(f'../data/train/output_2023_w{week:02d}.csv')], axis=0)
LOG.info(f'Tracking input shape: {tracking_input.shape}, output shape: {tracking_output.shape}')

games, plays, players, tracking = preprocess.process_data(tracking_input, tracking_output, sup_data)
plays['yards_to_goal'] = 110 - plays['absolute_yardline_number']
team_desc = preprocess.fetch_team_desc()

##############  ii. Predict if defender is a part of the pass play ##############
ds = DefenderReachDataset()
defender_df = ds.generate_defender_data(tracking, plays)
model = DefenderReachModel()
if not model.check_is_trained():
    raise Exception("Defender Reach Model not trained. Run train_gnn_epa_model.py before running this script.")
model.load()
defender_df.loc[:, ['within_10_yards_proba', 'within_10_yards_pred']] = model.predict(defender_df)

defender_df = (defender_df
    .query('within_10_yards_pred==1')
    .reset_index(drop=True)
    .merge(
        players[['nfl_id','player_position']],
        on='nfl_id',
        how='left'
    )
)

gpids_with_over_4_defenders = (
    defender_df
        .drop_duplicates(subset=['gpid','nfl_id'])
        .groupby(['gpid']).size()
        .reset_index()
        .rename(columns={0:'n_defenders_involved'})
        .sort_values('n_defenders_involved', ascending=False)
        .query('n_defenders_involved > 4')
).gpid.unique()

# Drop the defender with the lowest probability of being within 10 yards on plays with over 4 defenders until 4 defenders remain
for gpid in tqdm(gpids_with_over_4_defenders, desc="Filtering plays with over 4 defenders"):
    df = defender_df.query('gpid==@gpid').copy()
    while df.shape[0] > 4:
        min_proba_idx = df['within_10_yards_proba'].idxmin()
        defender_df.drop(index=min_proba_idx, inplace=True)
        df = defender_df.query('gpid==@gpid').copy()

# Drop plays with no defenders within 10 yards of the pass landing point
gpids_with_no_defenders = set(plays['gpid'].unique()) - set(defender_df['gpid'].unique())
if gpids_with_no_defenders:
    LOG.info(f"Dropping {len(gpids_with_no_defenders)} plays with no defenders within 10 yards of the pass landing point")
    defender_df = defender_df[~defender_df['gpid'].isin(gpids_with_no_defenders)].copy()

# Drop play if "S","FS","SS" not in defender positions
valid_positions = {'S', 'FS', 'SS'}
gpids_to_drop = []
for gpid in defender_df['gpid'].unique():
    positions = set(defender_df.query('gpid==@gpid')['player_position'].unique())
    if not positions.intersection(valid_positions):
        gpids_to_drop.append(gpid)
if gpids_to_drop:
    LOG.info(f"Dropping {len(gpids_to_drop)} plays with no safeties among defenders within 10 yards of the pass landing point")
    defender_df = defender_df[~defender_df['gpid'].isin(gpids_to_drop)].copy()

# Filters:
    # Safety must be in a free safety like position: at least 8 yards down field and within 10 yards of center field at time of snap
    # Pass distance must be at least 10 yards
    # Throw cannot be a "throw away": ball lands further than 5 yards from targeted receiver and qb threw outside of pocket
qb_outside_pocket = (
    tracking
    .query('pass_thrown and player_role=="Passer"')
    .sort_values('frame_id')
    .drop_duplicates(subset=['gpid'])
    .rename(columns={'y':'qb_y_at_throw'})
    .merge(
        tracking.query('position=="Ball" and frame_id==1').rename(columns={'y':'ball_y_at_snap'})[['gpid','ball_y_at_snap']],
        on='gpid',
        how='left'
    )
    .assign(
        qb_outside_pocket=lambda x: np.abs(x['qb_y_at_throw'] - x['ball_y_at_snap']) > 7
    )
)
ball_receiver_distance = (
    tracking
    .sort_values('frame_id', ascending=False)
    .query('player_role=="Targeted Receiver"')
    [['gpid','x','y']]
    .rename(columns={'x':'receiver_x','y':'receiver_y'})
    .drop_duplicates(subset=['gpid'])
    .merge(
        tracking
        .sort_values('frame_id', ascending=False)
        .query('position=="Ball"')
        [['gpid','x','y']]
        .rename(columns={'x':'ball_x','y':'ball_y'})
        .drop_duplicates(subset=['gpid']),
        on='gpid',
        how='left'
    )
    .assign(
        ball_receiver_distance=lambda x: np.sqrt((x['ball_x'] - x['receiver_x'])**2 + (x['ball_y'] - x['receiver_y'])**2)
    )
    [['gpid','ball_receiver_distance']]
)
free_safties = (
    tracking.merge(
        defender_df[['gpid','nfl_id']],
        on=['gpid','nfl_id'],
        how='inner'
    )
    .query('frame_id == 1 and position in @valid_positions')
    .merge(
        plays[['gpid','absolute_yardline_number', 'pass_distance']],
        on='gpid',
        how='left'
    )
    .assign(
        downfield_distance=lambda x: x['x'] - x['absolute_yardline_number'],
        lateral_distance=lambda x: np.abs(x['y'] - 26.65)
    )
    .query('downfield_distance >= 8 and lateral_distance <= 10')
    .query('pass_distance >= 10')
    .merge(
        qb_outside_pocket[['gpid','qb_outside_pocket']],
        on='gpid',
        how='left'
    )
    .assign(
        qb_outside_pocket=lambda x: x['qb_outside_pocket'].fillna(False)
    )
    .merge(
        ball_receiver_distance[['gpid','ball_receiver_distance']],
        on='gpid',
        how='left'
    )
    .assign(
        is_throw_away=lambda x: (x['ball_receiver_distance'] > 5) & x['qb_outside_pocket']
    )
    .query('~is_throw_away')
    .reset_index(drop=True)
    [['gpid','nfl_id']]
)
LOG.info(f"Number of free safeties plays after filtering: {free_safties['gpid'].nunique()}")
LOG.info(f'Number of unique free safeties: {free_safties["nfl_id"].nunique()}')

# Find the safety coordinates one second before the throw
safety_starting_coords = (
    tracking.merge(
        free_safties,
        on=['gpid','nfl_id'],
        how='inner'
    )
    .merge(
        tracking
            .query('pass_thrown')
            [['gpid','frame_id']]
            .sort_values('frame_id', ascending=True)
            .drop_duplicates(subset=['gpid'], keep='first')
            .rename(columns={'frame_id':'throw_frame_id'}),
        on='gpid',
        how='left'
    )
    .assign(one_sec_before_throw_frame_id=lambda x: np.maximum(x['throw_frame_id'] - 10, 1))
    .query('frame_id == one_sec_before_throw_frame_id')
    .assign(
        vx=lambda x: x['s'] * np.cos(np.deg2rad(x['dir'])),
        vy=lambda x: x['s'] * np.sin(np.deg2rad(x['dir']))
    )
    .rename(columns={
        'x':'safety_start_x',
        'y':'safety_start_y',
        'dir':'safety_start_dir',
        's':'safety_start_s',
        'vx':'safety_start_vx',
        'vy':'safety_start_vy'
    })
    .merge(
        plays[['gpid','num_frames_output']],
        on='gpid',
        how='left'
    )
    .assign(
        pass_time=lambda x: (x['num_frames_output'] / 10.0) + 1.0  # Time from 1 second before throw to ball arrival
    )
    [['gpid','nfl_id','safety_start_x','safety_start_y','safety_start_dir','safety_start_s',
      'safety_start_vx','safety_start_vy','pass_time']]
    .reset_index(drop=True)
)


##############  iii. Generate reachable points for each safety ##############
##############  iii. Generate reachable points for each safety ##############
def generate_safety_reachable_points(safety_df, grid_spacing=0.5, simulation_params={}):
    """
    Generate grid of reachable points for each safety.
    
    Args:
        safety_df (DataFrame): DataFrame with safety starting coordinates
        grid_spacing (float): Spacing for interior grid points in yards
        simulation_params (dict): Optional dictionary to override default simulation parameters
    
    Returns:
        DataFrame: Each row is a reachable point with columns:
                   gpid, nfl_id, point_type, safety_sim_x, safety_sim_y, 
                   safety_sim_dir, safety_sim_s, angle_rad,
                   safety_start_x, safety_start_y, safety_start_dir, safety_start_s
    """
    all_points = []
    
    for idx, row in tqdm(safety_df.iterrows(), total=len(safety_df), desc="Generating reachable points"):
        gpid = row['gpid']
        nfl_id = row['nfl_id']
        
        # Convert direction and speed to velocity vector
        init_pos = np.array([row['safety_start_x'], row['safety_start_y']])
        init_vel = np.array([row['safety_start_vx'], row['safety_start_vy']])
        
        # Simulate outer boundary points and get final velocities
        outer_points, final_vels, angles = simulate_outer_points(
            init_pos=init_pos,
            init_vel=init_vel,
            sim_time=row['pass_time'],
            **simulation_params
        )
        
        # Calculate direction and speed for boundary points
        # Direction: from starting point to boundary point
        boundary_dirs = np.arctan2(
            outer_points[:, 1] - init_pos[1],
            outer_points[:, 0] - init_pos[0]
        )
        
        # Convert from radians to degrees (0-360, where 0 = east, 90 = north)
        boundary_dirs_deg = np.degrees(boundary_dirs) % 360
        
        # Speed at boundary points (magnitude of final velocity)
        boundary_speeds = np.linalg.norm(final_vels, axis=1)
        
        # Generate interior grid points
        interior_points, _, _ = fill_polygon_with_grid(
            outer_pts=outer_points,
            spacing=grid_spacing
        )
        
        # Create DataFrame for boundary points
        boundary_df = pd.DataFrame({
            'gpid': gpid,
            'nfl_id': nfl_id,
            'point_type': 'boundary',
            'safety_sim_x': outer_points[:, 0],
            'safety_sim_y': outer_points[:, 1],
            'safety_sim_dir': boundary_dirs_deg,  # Direction from start to point (degrees)
            'safety_sim_s': boundary_speeds,      # Speed at point (yd/s)
            'angle_rad': angles,                   # Target angle in simulation
            'final_vel_x': final_vels[:, 0],       # Final velocity x-component
            'final_vel_y': final_vels[:, 1],       # Final velocity y-component
            'safety_start_x': row['safety_start_x'],
            'safety_start_y': row['safety_start_y'],
            'safety_start_dir': row['safety_start_dir'],
            'safety_start_s': row['safety_start_s']
        })
        
        # Create DataFrame for interior points
        if len(interior_points) > 0:
            # For interior points, we need to estimate direction and speed
            # Direction: from starting point to interior point
            interior_dirs = np.arctan2(
                interior_points[:, 1] - init_pos[1],
                interior_points[:, 0] - init_pos[0]
            )
            interior_dirs_deg = np.degrees(interior_dirs) % 360
            
            # For interior points, we need to estimate the speed
            # We can approximate by finding the closest boundary point and using its velocity
            # Or use a simpler approximation based on distance
            distances_to_start = np.linalg.norm(interior_points - init_pos, axis=1)
            max_distance = np.max(distances_to_start)
            
            # Simple speed estimate: assuming linear acceleration from start to point
            # This is an approximation since interior points may not be on optimal path
            time_to_point = row['pass_time']  # Using full pass time
            interior_speeds = distances_to_start / time_to_point
            
            # Cap at max speed
            max_speed = simulation_params.get('max_speed', 7.0)
            interior_speeds = np.minimum(interior_speeds, max_speed)
            
            interior_df = pd.DataFrame({
                'gpid': gpid,
                'nfl_id': nfl_id,
                'point_type': 'interior',
                'safety_sim_x': interior_points[:, 0],
                'safety_sim_y': interior_points[:, 1],
                'safety_sim_dir': interior_dirs_deg,  # Direction from start to point
                'safety_sim_s': interior_speeds,      # Estimated speed at point
                'angle_rad': np.nan,
                'final_vel_x': np.nan,                # Not available for interior points
                'final_vel_y': np.nan,
                'safety_start_x': row['safety_start_x'],
                'safety_start_y': row['safety_start_y'],
                'safety_start_dir': row['safety_start_dir'],
                'safety_start_s': row['safety_start_s']
            })
        else:
            interior_df = pd.DataFrame()
        
        # Combine boundary and interior points
        play_points = pd.concat([boundary_df, interior_df], ignore_index=True)
        all_points.append(play_points)
    
    # Combine all plays
    if all_points:
        all_points_df = pd.concat(all_points, ignore_index=True)
    else:
        all_points_df = pd.DataFrame(columns=[
            'gpid', 'nfl_id', 'point_type', 'safety_sim_x', 'safety_sim_y',
            'safety_sim_dir', 'safety_sim_s', 'angle_rad',
            'final_vel_x', 'final_vel_y',
            'safety_start_x', 'safety_start_y', 'safety_start_dir', 'safety_start_s'
        ])
    
    return all_points_df

# Generate reachable points for all safeties
LOG.info("Generating reachable points for safeties")
safety_reachable_points = generate_safety_reachable_points(
    safety_starting_coords,
    grid_spacing=0.5,
)

LOG.info(f"Generated {len(safety_reachable_points)} reachable points")
LOG.info(f"Points per play: {safety_reachable_points.groupby('gpid').size().describe()}")

##############  iii. Filter tracking data ###############
gpids = defender_df['gpid'].unique()
defender_gpid_nflids = set(defender_df['gpid'] + '_' + defender_df['nfl_id'].astype(str))
safety_gpids = free_safties['gpid'].unique()
df = (
    tracking
      .query('gpid in @gpids and gpid in @safety_gpids')
      .assign(gpid_nflid=lambda x: x['gpid'] + '_' + x['nfl_id'].astype(str))
      .query('gpid_nflid in @defender_gpid_nflids or position=="Ball" or player_role=="Targeted Receiver"')
      .merge(
          plays[['gpid','absolute_yardline_number','ball_land_x','ball_land_y','team_coverage_man_zone']],
          on='gpid',
          how='left'
      )
      # Normalize x coordinates relative to line of scrimmage
      .assign(
          x=lambda x: x['x'] - x['absolute_yardline_number'],
          ball_land_x=lambda x: x['ball_land_x'] - x['absolute_yardline_number'],
          zone_coverage=lambda x: np.where(x['team_coverage_man_zone'] == "ZONE_COVERAGE", 1, 0)
      )
      # Only keep last frame of play when ball lands
      .merge(
          tracking
            .query('pass_thrown')
            .groupby('gpid')['frame_id'].max()
            .reset_index()
            .rename(columns={'frame_id':'ball_land_frame_id'})
            [['gpid','ball_land_frame_id']],
          on='gpid',
          how='left'
      )
      .query('frame_id == ball_land_frame_id')
      .merge(
          defender_df[['gpid','nfl_id','within_10_yards_proba']],
          on=['gpid','nfl_id'],
          how='left'
      )
      [['gpid', 'frame_id', 'nfl_id', 'player_role', 'position', 'x', 'y', 's', 'dir',
        'ball_land_x','ball_land_y','zone_coverage','within_10_yards_proba']]
)

LOG.info(f"Final number of pass plays: {defender_df['gpid'].nunique()}")

# Join the actual safety positions to the reachable points dataframe
df = pd.concat([
    df.merge(
        free_safties.rename(columns={'nfl_id':'safety_nfl_id'}),
        on=['gpid'],
        how='left'
    )
    .assign(
        sample_type='actual',
    ),
    df.merge(
        safety_reachable_points
            [['gpid','nfl_id','safety_sim_x','safety_sim_y','safety_sim_dir','safety_sim_s']]
            .assign(sample_type='simulated')
            .rename(columns={'nfl_id':'safety_nfl_id'}),
        on=['gpid'],
        how='left'
    )
    .assign(
        x=lambda x: np.where(x.nfl_id == x.safety_nfl_id, x.safety_sim_x, x.x),
        y=lambda x: np.where(x.nfl_id == x.safety_nfl_id, x.safety_sim_y, x.y),
        s=lambda x: np.where(x.nfl_id == x.safety_nfl_id, x.safety_sim_s, x.s),
        dir=lambda x: np.where(x.nfl_id == x.safety_nfl_id, x.safety_sim_dir, x.dir)
    )
    .drop(columns=['safety_sim_x','safety_sim_y','safety_sim_dir','safety_sim_s'])
], ignore_index=True
).assign(key=lambda x: x.groupby(['gpid','safety_nfl_id','sample_type']).ngroup())

##############  iv. EPA Model Prediction ###############
EPA_MODEL_PATH = '/Users/lukeneuendorf/projects/nfl-big-data-bowl-2026/data/models/epa_gnn_model.pth'
from models.epa.epa_gnn_model import EPAGNN
epa_model = EPAGNN(
    node_feat_dim=4,
    node_type_count=3,
    edge_feat_dim=4,
    global_dim=5,
    hidden=128
)
checkpoint = torch.load(EPA_MODEL_PATH, map_location=torch.device('cpu'))
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    epa_model.load_state_dict(checkpoint['model_state_dict'])
else:
    epa_model.load_state_dict(checkpoint)
epa_model.eval()
epa_preds = pd.DataFrame(columns=['gpid','safety_nfl_id','pred_epa'])
for gpid, safety_nfl_id, sample_type, key in tqdm(df[['gpid', 'safety_nfl_id', 'sample_type', 'key']].drop_duplicates().itertuples(index=False), desc="Preparing samples for EPA model", total=df[['gpid', 'safety_nfl_id', 'sample_type', 'key']].drop_duplicates().shape[0]):
    play_tracking = df.query('gpid==@gpid and key==@key and sample_type==@sample_type').copy()
    play = plays.query('gpid==@gpid').iloc[0]
    assert play_tracking['frame_id'].nunique() == 1, f"More than one frame_id for gpid {gpid}"
    receiver_row = play_tracking.query('player_role=="Targeted Receiver"').iloc[0]
    ball_row = play_tracking.query('position=="Ball"').iloc[0]
    defender_rows = (
        play_tracking
          .query('player_role=="Defensive Coverage"')
          .sort_values('within_10_yards_proba', ascending=False).copy()
    )

    receiver = {
        'x': receiver_row['x'],
        'y': receiver_row['y'],
        'vx': receiver_row['s'] * np.cos(np.deg2rad(receiver_row['dir'])),
        'vy': receiver_row['s'] * np.sin(np.deg2rad(receiver_row['dir']))
    }
    ball = {
        'x': ball_row['x'],
        'y': ball_row['y'],
    }
    defenders = []
    for _, row in defender_rows.iterrows():
        defenders.append({
            'x': row['x'],
            'y': row['y'],
            'vx': row['s'] * np.cos(np.deg2rad(row['dir'])),
            'vy': row['s'] * np.sin(np.deg2rad(row['dir']))
        })
    if len(defenders) < 1:
        raise ValueError(f"No defenders for gpid {gpid}")
    global_features = {
        'zone_coverage': play_tracking['zone_coverage'].iloc[0],
        'down': play['down'],
        'ball_land_yards_to_first_down': max(play['yards_to_go'] - ball_row['x'], 0),
        'ball_land_yards_to_endzone': min(110 - (play['absolute_yardline_number'] + ball_row['x']), 0),
        'pass_distance': play['pass_distance']
    }
    target_epa = play['expected_points_added']

    sample = {
        'receiver': receiver,
        'ball': ball,
        'defenders': defenders,
        'global_features': global_features,
        'target_epa': target_epa
    }

    dataset = EPAGraphDataset([sample])
    graph = dataset[0]
    with torch.no_grad():
        pred_epa = epa_model(graph.x, graph.edge_index, graph.global_features.unsqueeze(0))
    epa_preds = pd.concat([
        epa_preds,
        pd.DataFrame({
            'gpid': [gpid],
            'safety_nfl_id': [safety_nfl_id],
            'sample_type': [sample_type],
            'key': [key],
            'pred_epa': pred_epa.squeeze().item()
        })
    ], ignore_index=True)

breakpoint()
print(1)
##############  v. Save results ###############
