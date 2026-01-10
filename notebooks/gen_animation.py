#!/usr/bin/env python
import logging
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

sys.path.append('../py')
from preprocess import preprocess
from nflplotlib import nflplot as nfp

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# N_WEEKS = 18  
# gpid = '2024010711_1919'
# safety_nfl_id = 52444
week = 15
gpid = '2023121602_115'
safety_nfl_id = 43387
FIELD_ALPHA = 0.5

game_id = int(gpid.split('_')[0])
play_id = int(gpid.split('_')[1])

##############  Load and preprocess the tracking + play data ##############
sup_data = pd.read_csv('../data/supplementary_data.csv')
tracking_input, tracking_output = pd.DataFrame(), pd.DataFrame()

tracking_input = (
    pd.concat([tracking_input, pd.read_csv(f'../data/train/input_2023_w{week:02d}.csv')], axis=0)
    .query("game_id == @game_id and play_id == @play_id")
)
tracking_output = (
    pd.concat([tracking_output, pd.read_csv(f'../data/train/output_2023_w{week:02d}.csv')], axis=0)
    .query("game_id == @game_id and play_id == @play_id")
)
games, plays, players, tracking = preprocess.process_data(tracking_input, tracking_output, sup_data)
team_desc = preprocess.fetch_team_desc()

# flip play direction to be right to left
plays = plays.assign(
    absolute_yardline_number = 120 - plays['absolute_yardline_number'],
    ball_land_x = 120 - plays['ball_land_x'],
    ball_land_y = 53.3 - plays['ball_land_y'] 
)
tracking = tracking.assign(
    x = 120 - tracking['x'],
    y = 53.3 - tracking['y'],
    dir = (tracking['dir'] + 180) % 360,
    o = (tracking['o'] + 180) % 360
)

# -----------------------------
# Animation
# -----------------------------
nfp.animate_play(
    tracking.query('gpid==@gpid'),
    plays.query('gpid==@gpid'),
    games.query(f'game_id=={gpid.split("_")[0]}'),
    team_desc,
    save_path='/Users/lukeneuendorf/projects/nfl-big-data-bowl-2026/animation.mp4',
    plot_positions=True,
    plot_arrows=True,
    show_ball_trajectory=True,
    plot_heatmap=True
)