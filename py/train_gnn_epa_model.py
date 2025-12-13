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
from models.epa.epa_gnn_model import train_model, evaluate_split


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
team_desc = preprocess.fetch_team_desc()

##############  ii. Predict if defender is a part of the pass play ##############
ds = DefenderReachDataset()
defender_df = ds.generate_defender_data(tracking, plays)
model = DefenderReachModel()
if not model.check_is_trained():
    model.train(defender_df, save_model=True)
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

##############  iii. Filter tracking data ###############
gpids = defender_df['gpid'].unique()
defender_gpid_nflids = set(defender_df['gpid'] + '_' + defender_df['nfl_id'].astype(str))
df = (
    tracking
      .query('gpid in @gpids')
      .assign(gpid_nflid=lambda x: x['gpid'] + '_' + x['nfl_id'].astype(str))
      .query('gpid_nflid in @defender_gpid_nflids or position=="Ball" or player_role=="Targeted Receiver"')
      .merge(
          plays[['gpid','absolute_yardline_number','ball_land_x','ball_land_y','team_coverage_man_zone','route_of_targeted_receiver']],
          on='gpid',
          how='left'
      )
      # Normalize x coordinates relative to line of scrimmage
      .assign(
          x=lambda x: x['x'] - x['absolute_yardline_number'],
          ball_land_x=lambda x: x['ball_land_x'] - x['absolute_yardline_number'],
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
)

LOG.info(f"Final number of pass plays: {defender_df['gpid'].nunique()}")

##############  iv. EPA Model Training ###############

# Training games week 1-14, validation week 15-16, test week 17-18
train_games = games.query('week <= 14')['game_id'].unique()
val_games = games.query('week >= 15 and week <=16')['game_id'].unique()
test_games = games.query('week >= 17')['game_id'].unique()
train_samples, val_samples, test_samples = [], [], []
for gpid in tqdm(df['gpid'].unique(), desc="Preparing samples for EPA model"):
    play_tracking = df.query('gpid==@gpid').copy()
    play = plays.query('gpid==@gpid').iloc[0]
    absolute_yardline_number = play['absolute_yardline_number'].item()
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
        'vx': ball_row['s'] * np.cos(np.deg2rad(ball_row['dir'])),
        'vy': ball_row['s'] * np.sin(np.deg2rad(ball_row['dir']))
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
        'ball_land_yards_to_endzone': max(110 - (absolute_yardline_number + ball_row['x']), 0),
        'pass_distance': play['pass_distance'],
        'route_vertical': receiver_row['route_vertical'],
        'route_inside_break': receiver_row['route_inside_break'],
        'route_outside_break': receiver_row['route_outside_break'],
        'route_underneath_short': receiver_row['route_underneath_short']
    }
    target_epa = play['expected_points_added']

    sample = {
        'absolute_yardline_number': absolute_yardline_number,
        'receiver': receiver,
        'ball': ball,
        'defenders': defenders,
        'global_features': global_features,
        'target_epa': target_epa
    }
    game_id = int(gpid.split('_')[0])
    if game_id in train_games:
        train_samples.append(sample)
    elif game_id in val_games:
        val_samples.append(sample)
    elif game_id in test_games:
        test_samples.append(sample)
    else:
        raise ValueError(f"Game ID {game_id} not found in any split")
train_dataset = EPAGraphDataset(train_samples)
val_dataset = EPAGraphDataset(val_samples)
test_dataset = EPAGraphDataset(test_samples)

model = train_model(
    train_dataset, 
    val_dataset, 
    batch_size=32, 
    lr=1e-3, 
    epochs=100, 
    patience=5
)

# ------------------------------
# Evaluate Model
# ------------------------------
train_metrics = evaluate_split(model, train_dataset)
val_metrics   = evaluate_split(model, val_dataset)
test_metrics  = evaluate_split(model, test_dataset)

LOG.info("===== Final EPA Model Evaluation =====")
LOG.info(f"{'Split':<10} {'MAE':>10} {'SmoothL1':>12} {'MSE':>12} {'RMSE':>12}")
LOG.info("-" * 60)
LOG.info(f"{'Train':<10} {train_metrics['MAE']:10.4f} {train_metrics['SmoothL1']:12.4f} "
      f"{train_metrics['MSE']:12.4f} {train_metrics['RMSE']:12.4f}")
LOG.info(f"{'Val':<10} {val_metrics['MAE']:10.4f} {val_metrics['SmoothL1']:12.4f} "
      f"{val_metrics['MSE']:12.4f} {val_metrics['RMSE']:12.4f}")
LOG.info(f"{'Test':<10} {test_metrics['MAE']:10.4f} {test_metrics['SmoothL1']:12.4f} "
      f"{test_metrics['MSE']:12.4f} {test_metrics['RMSE']:12.4f}")

# Baseline: Predict average EPA
def smooth_l1(y_true, y_pred, delta=1.0):
    diff = np.abs(y_true - y_pred)
    loss = np.where(diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))
    return np.mean(loss)
avg_epa = plays.expected_points_added.mean()
LOG.info("===== Baseline EPA Model Evaluation (Predicting Average EPA) =====")
LOG.info(f"{'Split':<10} {'MAE':>10} {'SmoothL1':>12} {'MSE':>12} {'RMSE':>12}")
LOG.info("-" * 60)
LOG.info(f"{'All':<10} "
      f"{np.mean(np.abs(plays.expected_points_added - avg_epa)):10.4f} "
      f"{smooth_l1(plays.expected_points_added, avg_epa):12.4f} "
      f"{np.mean((plays.expected_points_added - avg_epa)**2):12.4f} "
      f"{np.sqrt(np.mean((plays.expected_points_added - avg_epa)**2)):12.4f}")

LOG.info("Model training and evaluation complete, saving the model")
SAVE_PATH = '/Users/lukeneuendorf/projects/nfl-big-data-bowl-2026/data/models/epa_gnn_model.pth'
torch.save(model.state_dict(), SAVE_PATH)