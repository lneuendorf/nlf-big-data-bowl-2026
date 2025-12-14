import logging
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Batch

from preprocess import preprocess
from models.defender_reach.preprocessor import DefenderReachDataset
from models.defender_reach.model import DefenderReachModel
from models.interception.graph_dataset import IntGraphDataset
from models.safety_reachable_points.safety_reach import (
    simulate_outer_points,
    fill_polygon_with_grid
)
from models.interception.int_gnn_model import IntGNN

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RANDOM_SEED = 2
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
N_WEEKS = 18
SAVE_CSV_PATH = '/Users/lukeneuendorf/projects/nfl-big-data-bowl-2026/data/results'

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
team_desc = preprocess.fetch_team_desc()

# Normalize the x coordinates relative to line of scrimmage
tracking = tracking.merge(
    plays[['gpid','absolute_yardline_number']],
    on='gpid',
    how='left'
).assign(
    x=lambda x: x['x'] - x['absolute_yardline_number']
)
plays['ball_land_x'] = plays['ball_land_x'] - plays['absolute_yardline_number']

##############  ii. Predict if defender is a part of the pass play ##############
ds = DefenderReachDataset()
defender_df = ds.generate_defender_data(tracking, plays)
model = DefenderReachModel()
if not model.check_is_trained():
    raise Exception("Defender Reach Model not trained. Run train_gnn_int_model.py before running this script.")
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
    .drop_duplicates(subset=['gpid'], keep='first')
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
        plays[['gpid', 'pass_distance']],
        on='gpid',
        how='left'
    )
    .assign(
        downfield_distance=lambda x: x['x'],
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
            'start_x': row['safety_start_x'],
            'start_y': row['safety_start_y'],
            'start_dir': row['safety_start_dir'],
            'start_s': row['safety_start_s']
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
                'start_x': row['safety_start_x'],
                'start_y': row['safety_start_y'],
                'start_dir': row['safety_start_dir'],
                'start_s': row['safety_start_s']
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
            'safety_sim_dir', 'safety_sim_s',
            'start_x', 'start_y', 'start_dir', 'start_s'
        ])
    
    return all_points_df

# Generate reachable points for all safeties
LOG.info("Generating reachable points for safeties")
safety_reachable_points = generate_safety_reachable_points(
    safety_starting_coords,
    grid_spacing=0.5,
).assign(
    x=lambda x: x['safety_sim_x'],
    y=lambda x: x['safety_sim_y'],
    vx=lambda x: x['safety_sim_s'] * np.cos(np.deg2rad(x['safety_sim_dir'])),
    vy=lambda x: x['safety_sim_s'] * np.sin(np.deg2rad(x['safety_sim_dir']))
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
          plays[['gpid','ball_land_x','ball_land_y','team_coverage_man_zone',
                 'route_of_targeted_receiver']],
          on='gpid',
          how='left'
      )
      .assign(
          zone_coverage=lambda x: np.where(x['team_coverage_man_zone'] == "ZONE_COVERAGE", 1, 0),
          route_type=lambda x: np.select(
                [
                    x['route_of_targeted_receiver'].isin(['GO', 'POST', 'CORNER', 'WHEEL']),
                    x['route_of_targeted_receiver'].isin(['IN', 'SLANT', 'CROSS', 'ANGLE']),
                    x['route_of_targeted_receiver'].isin(['OUT']),
                    x['route_of_targeted_receiver'].isin(['HITCH', 'FLAT', 'SCREEN'])
                ],
                [
                    'VERTICAL',
                    'INSIDE_BREAK',
                    'OUTSIDE_BREAK',
                    'UNDERNEATH_SHORT'
                ],
                default='OTHER'
            )
      )
      .assign(
        route_vertical=lambda x: np.where(x['route_type']=='VERTICAL', 1, 0),
        route_inside_break=lambda x: np.where(x['route_type']=='INSIDE_BREAK', 1, 0),
        route_outside_break=lambda x: np.where(x['route_type']=='OUTSIDE_BREAK', 1, 0),
        route_underneath_short=lambda x: np.where(x['route_type']=='UNDERNEATH_SHORT', 1, 0)
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
        'ball_land_x','ball_land_y','zone_coverage','within_10_yards_proba',
        'route_vertical','route_inside_break','route_outside_break','route_underneath_short']]
      .merge(
          plays[['gpid', 'yards_to_go', 'down', 'absolute_yardline_number', 'pass_distance']],
          on='gpid',
          how='left'
      )
      .sort_values(['gpid','player_role','within_10_yards_proba'], ascending=[True, True, False])
      .assign(
          vx=lambda x: x['s'] * np.cos(np.deg2rad(x['dir'])),
          vy=lambda x: x['s'] * np.sin(np.deg2rad(x['dir'])),
          ball_land_yards_to_first_down=lambda x: np.maximum(x['yards_to_go'] - x['ball_land_x'], 0),
          ball_land_yards_to_endzone=lambda x: np.maximum(110 - (x['absolute_yardline_number'] + x['ball_land_x']), 0)
      )
)

##############  iv. Generate the Graphs ###############
samples = []
meta_data = []
# For each play and each free safety, create samples for original and simulated points
for gpid, group in tqdm(df.groupby('gpid'), desc="Formatting samples for Int prediction"):
    safety_nfl_ids = free_safties.query('gpid==@gpid').nfl_id.unique().tolist()
    defenders = group.query('player_role=="Defensive Coverage"').reset_index(drop=True)
    defender_map = {row['nfl_id']: idx for idx, row in defenders.iterrows()}
    defenders_list = defenders[['x','y','vx','vy']].to_dict(orient='records')
    base_sample = {
        'absolute_yardline_number': plays.query('gpid==@gpid')['absolute_yardline_number'].iloc[0],
        'receiver': group.query('player_role=="Targeted Receiver"')[['x','y','vx','vy']].iloc[0].to_dict(),
        'ball': group.query('position=="Ball"')[['x','y','vx','vy']].iloc[0].to_dict(),
        'defenders': defenders_list,
        'global_features': group[['zone_coverage','pass_distance',
                                 'route_vertical','route_inside_break','route_outside_break','route_underneath_short']].iloc[0].to_dict(),
        'target_int': None  # Placeholder, as we are predicting Int
    }

    for safety_nfl_id in safety_nfl_ids:
        safety_points = safety_reachable_points.query('gpid==@gpid and nfl_id==@safety_nfl_id').copy()
        start_points = safety_points[['start_x','start_y','start_dir','start_s']].iloc[0].to_dict()

        # Original sample with actual safety position
        samples.append({
            'absolute_yardline_number': base_sample['absolute_yardline_number'].copy(),
            'receiver': base_sample['receiver'].copy(),
            'ball': base_sample['ball'].copy(),
            'defenders': [d.copy() for d in base_sample['defenders']],
            'global_features': base_sample['global_features'].copy(),
            'target_int': None
        })
        meta_data.append({
            'gpid': gpid,
            'safety_nfl_id': safety_nfl_id,
            'sample_type': 'original',
            **group.query('nfl_id==@safety_nfl_id')[['x','y','vx','vy']].iloc[0].to_dict(),
            **start_points
        })

        # Simulated samples for each reachable point
        for _, point_row in safety_points.iterrows():
            modified_sample = {
                'absolute_yardline_number': base_sample['absolute_yardline_number'].copy(),
                'receiver': base_sample['receiver'].copy(),
                'ball': base_sample['ball'].copy(),
                'defenders': [d.copy() for d in base_sample['defenders']],  # Deep copy the list of dicts
                'global_features': base_sample['global_features'].copy()
            }
            modified_sample['defenders'][defender_map[safety_nfl_id]] = point_row[['x','y','vx','vy']].to_dict()
            samples.append(modified_sample)
            meta_data.append({
                'gpid': gpid,
                'safety_nfl_id': safety_nfl_id, 
                'sample_type': 'simulated',
                **point_row[['x','y','vx','vy']].to_dict(),
                **start_points
            })
LOG.info(f"Total number of samples (original + simulated): {len(samples)}")

graph_dataset = IntGraphDataset(samples)

##############  v. Batch Predict INT ###############
INT_MODEL_PATH = '/Users/lukeneuendorf/projects/nfl-big-data-bowl-2026/data/models/int_gnn_model.pth'
int_model = IntGNN(
    node_feat_dim=7,
    node_type_count=3,
    edge_feat_dim=11,
    global_dim=6,
    hidden=64,
)
checkpoint = torch.load(INT_MODEL_PATH, map_location=torch.device('cpu'))
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    int_model.load_state_dict(checkpoint['model_state_dict'])
else:
    int_model.load_state_dict(checkpoint)
int_model.eval()
BATCH_SIZE = 64
all_int_predictions = []
for i in tqdm(range(0, graph_dataset.len(), BATCH_SIZE), desc="Predicting INT for samples"):
    batch_samples = [graph_dataset.get(j) for j in range(i, min(i + BATCH_SIZE, graph_dataset.len()))]
    batch = Batch.from_data_list(batch_samples)
    with torch.no_grad():
        int_logits = int_model(batch)
        int_preds = torch.sigmoid(int_logits).squeeze()
    if int_preds.dim() == 0:
        all_int_predictions.append(int_preds.item())
    else:
        all_int_predictions.extend(int_preds.cpu().numpy().flatten().tolist())

##############  vi. Save Results ###############
results_df = pd.DataFrame(meta_data)
results_df['predicted_int'] = all_int_predictions   
os.makedirs(SAVE_CSV_PATH, exist_ok=True)
results_df.to_csv(os.path.join(SAVE_CSV_PATH, 'int_preds.csv'), index=False)