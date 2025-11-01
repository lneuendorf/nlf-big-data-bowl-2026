import sys
import os
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

sys.path.append('../py')
import preprocess
from model_training.cnn import (
    format_data_for_cnn_training,
    make_spatial_grid,
    train_cnn
)

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------- Configurables ----------
N_WEEKS = 1
RANDOM_SEED = 2
np.random.seed(RANDOM_SEED)

# ---------- Load Data ----------
sup_data = pd.read_csv('../data/supplementary_data.csv')
tracking_input, tracking_output = pd.DataFrame(), pd.DataFrame()
for week in tqdm(range(1, N_WEEKS+1), desc="Loading weekly data"):
    tracking_input = pd.concat([tracking_input, pd.read_csv(f'../data/train/input_2023_w{week:02d}.csv')], axis=0)
    tracking_output = pd.concat([tracking_output, pd.read_csv(f'../data/train/output_2023_w{week:02d}.csv')], axis=0)
LOG.info(f'Tracking input shape: {tracking_input.shape}, output shape: {tracking_output.shape}')

# ---------- Preprocess Data ----------
LOG.info('Preprocessing tracking data...')
games, plays, players, tracking = preprocess.process_data(tracking_input, tracking_output, sup_data)

# ---------- Append Play-Level Features ----------
tracking = tracking.merge(
    plays[['gpid', 'expected_points_added', 'ball_land_x', 'ball_land_y']],
    on='gpid',
    how='left'
).rename(columns={'expected_points_added': 'epa'})

# ---------- Format Tracking Data ----------
LOG.info('Formatting data for CNN training...')
tracking = format_data_for_cnn_training(tracking)
tracking["s_x"] = tracking["s"] * np.cos(tracking["dir"])
tracking["s_y"] = tracking["s"] * np.sin(tracking["dir"])

# ---------- Convert Data to Grid Tensors ----------
LOG.info('Converting tracking data to spatial grid tensors...')
frames_list = []
epa_list = []
for (gpid, frame_id), frame_df in tracking.groupby(['gpid', 'frame_id']):
    frames_list.append(make_spatial_grid(frame_df))
    epa_list.append(frame_df['epa'].iloc[0])

# ---------- Train CNN Model ----------
LOG.info('Training CNN model...')
model, embeddings_np = train_cnn(frames_list, epa_list)

# ---------- Save Embeddings ----------
LOG.info('Saving CNN embeddings...')
os.mkdir('../data') if not os.path.exists('../data') else None
os.mkdir('../data/models') if not os.path.exists('../data/models') else None
np.save('../data/models/cnn_embeddings.npy', embeddings_np)