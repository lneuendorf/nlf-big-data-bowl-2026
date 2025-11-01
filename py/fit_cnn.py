import sys
import os
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch

sys.path.append('../py')
from preprocess import preprocess
from models.cnn import (
    format_data_for_cnn_training,
    make_spatial_grid,
    train_cnn,
    save_cnn_model
)

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------- Configurables ----------
N_WEEKS = 18 # up to 18 weeks of training data
USE_CACHED_GRID_DATA = True
RANDOM_SEED = 2
np.random.seed(RANDOM_SEED)

# Define cache file paths
cache_dir = "../data/cache"
os.makedirs(cache_dir, exist_ok=True)
gpid_cache = os.path.join(cache_dir, "gpid_list.npy")
epa_cache = os.path.join(cache_dir, "epa_list.npy")
frames_cache = os.path.join(cache_dir, "frames_list.pt")

# Check if cached files exist
if os.path.exists(gpid_cache) and os.path.exists(epa_cache) and os.path.exists(frames_cache) and USE_CACHED_GRID_DATA:
    LOG.info("Loading cached gpid_list, epa_list, and frames_list from disk...")
    gpid_list = np.load(gpid_cache, allow_pickle=True).tolist()
    epa_list = np.load(epa_cache, allow_pickle=True).tolist()
    frames_list = torch.load(frames_cache, weights_only=False)
else:
    # ---------- Load Data ----------
    sup_data = pd.read_csv('../data/supplementary_data.csv')
    tracking_input, tracking_output = pd.DataFrame(), pd.DataFrame()
    for week in tqdm(range(1, N_WEEKS+1), desc="Loading weekly data"):
        tracking_input = pd.concat([tracking_input, pd.read_csv(f'../data/train/input_2023_w{week:02d}.csv')], axis=0)
        tracking_output = pd.concat([tracking_output, pd.read_csv(f'../data/train/output_2023_w{week:02d}.csv')], axis=0)
    LOG.info(f'Tracking input shape: {tracking_input.shape}, output shape: {tracking_output.shape}')

    # ---------- Preprocess Data ----------
    LOG.info('Preprocessing tracking data')
    games, plays, players, tracking = preprocess.process_data(tracking_input, tracking_output, sup_data)

    # ---------- Append Play-Level Features ----------
    tracking = tracking.merge(
        plays[['gpid', 'expected_points_added', 'ball_land_x', 'ball_land_y']],
        on='gpid',
        how='left'
    ).rename(columns={'expected_points_added': 'epa'})

    # ---------- Format Tracking Data ----------
    LOG.info('Formatting data for CNN training')
    tracking = format_data_for_cnn_training(tracking)
    tracking["s_x"] = tracking["s"] * np.cos(np.radians(tracking["dir"]))
    tracking["s_y"] = tracking["s"] * np.sin(np.radians(tracking["dir"]))
    LOG.info("Generating and caching gpid_list, epa_list, and frames_list...")
    gpid_list, frames_list, epa_list = [], [], []
    n_groups = tracking.drop_duplicates(['gpid', 'frame_id']).shape[0]
    for (gpid, frame_id), frame_df in tqdm(tracking.groupby(['gpid', 'frame_id']), total=n_groups, desc="Processing frames"):
        gpid_list.append(gpid)
        frames_list.append(make_spatial_grid(frame_df))
        epa_list.append(frame_df['epa'].iloc[0])

    # Save cached versions for next run
    np.save(gpid_cache, np.array(gpid_list, dtype=object))
    np.save(epa_cache, np.array(epa_list, dtype=object))
    torch.save(frames_list, frames_cache)

# ---------- Train CNN Model ----------
LOG.info('Training CNN model')
model, embeddings_np = train_cnn(gpid_list, frames_list, epa_list)

# ---------- Save model weights ----------
LOG.info('Saving CNN model weights')
os.mkdir('../data') if not os.path.exists('../data') else None
os.mkdir('../data/models') if not os.path.exists('../data/models') else None
save_cnn_model(model, '../data/models/cnn_model_weights.pth')