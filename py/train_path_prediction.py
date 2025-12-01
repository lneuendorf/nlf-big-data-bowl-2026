import sys
import os
import logging
from tqdm import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import polars as pl
import numpy as np

from preprocess import preprocess
from nflplotlib import nflplot as nfp
from models.defender_reach.preprocessor import DefenderReachDataset
from models.defender_reach.model import DefenderReachModel
from models.path_prediction.preprocessor import PathPredictionDataset
from models.path_prediction.model import train_path_prediction, predict_path

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

USE_TRAINED_PATH_PREDICTION_MODEL = False

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RANDOM_SEED = 2
np.random.seed(RANDOM_SEED)
N_WEEKS = 18

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

if 'player_position' in defender_df.columns:
    defender_df = defender_df.drop(columns=['player_position'])
    
defender_df = (defender_df
    .query('within_10_yards_pred==1')
    .reset_index(drop=True)
    .merge(
        players[['nfl_id','player_position']],
        on='nfl_id',
        how='left'
    )
)

gpids_with_over_5_defenders = (
    defender_df
        .drop_duplicates(subset=['gpid','nfl_id'])
        .groupby(['gpid']).size()
        .reset_index()
        .rename(columns={0:'n_defenders_involved'})
        .sort_values('n_defenders_involved', ascending=False)
        .query('n_defenders_involved > 5')
).gpid.unique()

# Drop the defender with the lowest probability of being within 10 yards on plays with over 5 defenders until 5 defenders remain
for gpid in tqdm(gpids_with_over_5_defenders, desc="Filtering plays with over 5 defenders"):
    df = defender_df.query('gpid==@gpid').copy()
    while df.shape[0] > 5:
        min_proba_idx = df['within_10_yards_proba'].idxmin()
        defender_df.drop(index=min_proba_idx, inplace=True)
        df = defender_df.query('gpid==@gpid').copy()
    
    # if gpid no longer has a safety, drop the gpid
    if not defender_df.query('gpid==@gpid and player_position.isin(["FS","SS","S"])').shape[0]:
        LOG.info(f"Dropping gpid {gpid} as it no longer has a safety")
        defender_df = defender_df.query('gpid!=@gpid').copy()

##############  iii. Filter tracking data ###############
gpids = defender_df['gpid'].unique()
gpid_nflids = set(defender_df['gpid'] + '_' + defender_df['nfl_id'].astype(str))
df = (
    tracking
      .query('gpid in @gpids')
      .assign(gpid_nflid=lambda x: x['gpid'] + '_' + x['nfl_id'].astype(str))
      .query('gpid_nflid in @gpid_nflids or position=="Ball" or player_role=="Targeted Receiver"')
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
      # Only keep frames within 1 second before pass is thrown and after
      .merge(
          tracking
            .query('pass_thrown')
            .groupby('gpid')['frame_id'].min()
            .reset_index()
            .rename(columns={'frame_id':'pass_thrown_frame_id'})
            [['gpid','pass_thrown_frame_id']],
          on='gpid',
          how='left'
      )
      .assign(
          one_second_before_pass_frame_id=lambda x: x['pass_thrown_frame_id'] - 10
      )
      .query('frame_id >= one_second_before_pass_frame_id')
      .merge(
          defender_df[['gpid','nfl_id','within_10_yards_proba']],
          on=['gpid','nfl_id'],
          how='left'
      )
      [['gpid', 'frame_id', 'nfl_id', 'player_role', 'position', 'x', 'y', 's', 'dir',
        'ball_land_x','ball_land_y','zone_coverage','within_10_yards_proba']]
)

##############  iv. Predict paths of defenders ##############
processed_plays = PathPredictionDataset().process(df)
if not os.path.exists('../data/models/path_prediction/best_model.pth') or not USE_TRAINED_PATH_PREDICTION_MODEL:
    train_path_prediction(processed_plays)
predicted_paths = predict_path(processed_plays)
predicted_paths = (
    predicted_paths.merge(
        df[['gpid', 'frame_id', 'nfl_id', 'x', 'y']].rename(columns={'x':'true_x','y':'true_y'}),
        on=['gpid','frame_id','nfl_id'],
        how='left'
    )
)
Path( '../data/results/').mkdir(parents=True, exist_ok=True)
predicted_paths.to_csv('../data/results/path_prediction_results.csv', index=False)