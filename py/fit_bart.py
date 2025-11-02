import sys
import os
from pathlib import Path
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

import pymc as pm
import pymc_bart as pmb
import arviz as az

sys.path.append('../py')
from preprocess import preprocess
from models.cnn import (
    format_data_for_cnn_training,
    make_spatial_grid,
    load_cnn_model
)

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
OUT_DIR = Path("pmb_outputs")
OUT_DIR.mkdir(exist_ok=True)
USE_CACHED_GRID_DATA = True
SAVE_TRACE = False
N_CHAINS = 4
N_TUNE = 1000
N_DRAW = 1000
BART_TREES = 50   # m in pmb.BART
N_WEEKS = 18
RANDOM_SEED = 2
np.random.seed(RANDOM_SEED)

# Define cache file paths
cache_dir = "../data/cache"
os.makedirs(cache_dir, exist_ok=True)
gpid_cache = os.path.join(cache_dir, "gpid_list.npy")
frame_id_cache = os.path.join(cache_dir, "frames_id_list.npy")
epa_cache = os.path.join(cache_dir, "epa_list.npy")
frames_cache = os.path.join(cache_dir, "frames_list.pt")

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

# Check if cached files exist
if os.path.exists(gpid_cache) and os.path.exists(frame_id_cache) and os.path.exists(epa_cache) and os.path.exists(frames_cache) and USE_CACHED_GRID_DATA:
    LOG.info("Loading cached gpid_list, epa_list, and frames_list from disk...")
    gpid_list = np.load(gpid_cache, allow_pickle=True).tolist()
    frame_id_list = np.load(frame_id_cache, allow_pickle=True).tolist()
    epa_list = np.load(epa_cache, allow_pickle=True).tolist()
    frames_list = torch.load(frames_cache, weights_only=False)
else:
    # ---------- Convert Data to Grid Tensors ----------
    LOG.info('Converting tracking data to spatial grid tensors')
    gpid_list, frames_list, frame_id_list, epa_list = [], [], [], []
    n_groups = tracking.drop_duplicates(['gpid', 'frame_id']).shape[0]
    for (gpid, frame_id), frame_df in tqdm(tracking.groupby(['gpid', 'frame_id']), total=n_groups, desc="Processing frames"):
        gpid_list.append(gpid)
        frame_id_list.append(frame_id)
        frames_list.append(make_spatial_grid(frame_df))
        epa_list.append(frame_df['epa'].iloc[0])

    # Save cached versions for next run
    np.save(gpid_cache, np.array(gpid_list, dtype=object))
    np.save(epa_cache, np.array(epa_list, dtype=object))
    np.save(frame_id_cache, np.array(frame_id_list, dtype=object))
    torch.save(frames_list, frames_cache)

# --- Generate CNN Embeddings (batched) ---
LOG.info('Generating CNN embeddings')
model = load_cnn_model(in_channels=frames_list[0].shape[0])
model.eval()

batch_size = 64
embeddings_list, epa_preds_list = [], []

with torch.no_grad():
    for i in tqdm(range(0, len(frames_list), batch_size), desc="Generating embeddings"):
        # Use torch.stack to create a batch of tensors
        batch = torch.stack([torch.tensor(frame, dtype=torch.float32) for frame in frames_list[i:i+batch_size]], dim=0)
        epa_pred, emb  = model(batch)
        embeddings_list.append(emb.cpu())
        epa_preds_list.append(epa_pred.cpu())

embeddings_np = torch.cat(embeddings_list).numpy()
epa_preds_np = torch.cat(epa_preds_list).numpy()

# --- Combine metadata ---
embeddings_df = pd.DataFrame({
    'gpid': gpid_list,
    'frame_id': frame_id_list,
    'epa': epa_list
})
embeddings_df = pd.concat([embeddings_df, pd.DataFrame(embeddings_np)], axis=1)

# rename 0,1,... to embedding_0, embedding_1,...
embeddings_df.rename(columns={
    i: f"embedding_{i}" for i in range(embeddings_np.shape[1])
}, inplace=True)

def get_ball_flight_pct(df):
    """
    Compute percent of ball flight for all plays in one pass (vectorized).
    Returns a copy with new 'ball_flight_pct' column.
    """
    df = df.sort_values(['gpid','frame_id']).copy()

    # Find throw frames and end frames per play
    throw_frame = (
        df.loc[df['pass_thrown'], ['gpid','frame_id']]
        .groupby('gpid')['frame_id']
        .min()
        .rename('throw_frame')
    )
    end_frame = df.groupby('gpid')['frame_id'].max().rename('end_frame')

    df = df.merge(throw_frame, on='gpid', how='left').merge(end_frame, on='gpid', how='left')

    # Compute flight pct
    df['ball_flight_pct'] = 0.0
    in_flight = df['frame_id'] >= df['throw_frame']
    df.loc[in_flight, 'ball_flight_pct'] = (
        (df.loc[in_flight, 'frame_id'] - df.loc[in_flight, 'throw_frame'])
        / (df.loc[in_flight, 'end_frame'] - df.loc[in_flight, 'throw_frame']).clip(lower=1)
    ) * 100

    return df.drop(columns=['throw_frame','end_frame'])

def prepare_xy(embeddings_df):
    """Given embeddings_df, return X (np.ndarray) and Y (np.ndarray) and feature names list."""
    embed_cols = [f"embedding_{i}" for i in range(32)]   # embedding_0 ... embedding_31
    feature_cols = embed_cols + ["ball_flight_pct", "down_x_dist", "pass_length", "pass_dist_to_goal", "ball_land_dist_from_sideline"]
    missing = [c for c in feature_cols if c not in embeddings_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in embeddings_df: {missing}")

    X = embeddings_df[feature_cols].astype(float).to_numpy()
    Y = embeddings_df["epa"].astype(float).to_numpy()
    return X, Y, feature_cols

# -----------------------------------------------------------------------------
# MODEL FITTING
# -----------------------------------------------------------------------------
def fit_bart_model(X, Y, feature_names=None, out_dir=OUT_DIR):
    """Fit a BART model to predict Y from X and save diagnostics and trace."""
    rng = np.random.default_rng(RANDOM_SEED)
    n_obs, n_features = X.shape
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]

    with pm.Model() as model:
        # optional non-BART variable (intercept/scale) — keeps diagnostics examples consistent
        α = pm.Normal("α", mu=0.0, sigma=1.0)

        # BART random variable (the flexible function)
        # pmb.BART(name, X, Y, m=number_of_trees)
        μ_bart = pmb.BART("μ_bart", X=X, Y=Y, m=BART_TREES)

        # Combine intercept + BART mean if you want (here we add intercept for clarity)
        μ = pm.Deterministic("μ", α + μ_bart)

        # Observation noise
        σ = pm.HalfNormal("σ", sigma=np.std(Y) if np.std(Y) > 0 else 1.0)

        # Likelihood
        y_obs = pm.Normal("y_obs", mu=μ, sigma=σ, observed=Y)

        # SAMPLE
        idata = pm.sample(draws=N_DRAW, tune=N_TUNE, chains=N_CHAINS,
                          random_seed=RANDOM_SEED, target_accept=0.95)

    # Save trace to NetCDF
    if SAVE_TRACE:
        trace_path = out_dir / "pmb_bart_epa_trace.nc"
        az.to_netcdf(idata, trace_path)
        LOG.info(f"Saved trace to {trace_path}")

    # ---------- Diagnostics & plots ----------
    # 1) Trace plot for α
    fig, axes = plt.subplots()  # Create a figure manually
    axes = az.plot_trace(idata, var_names=["α"], kind="rank_bars")
    fig = axes.ravel()[0].figure  # Get the figure from the first axis
    fig.suptitle("Trace (α) - rank_bars")
    plt.tight_layout()
    fpath = out_dir / "trace_alpha_rank_bars.png"
    plt.savefig(fpath, dpi=150)
    plt.close(fig)
    LOG.info(f"Saved: {fpath}")

    # 2) PyMC-BART convergence plot for the BART variable μ
    # fig = pmb.plot_convergence(idata, var_name="μ")
    # fpath = out_dir / "pmb_plot_convergence_mu.png"
    # fig.figure.savefig(fpath, dpi=150)
    # plt.close(fig.figure)
    # LOG.info(f"Saved: {fpath}")

    # 3) PDP (partial dependence) plots
    # pmb.plot_pdp expects the BART random variable object (μ_bart) and X, Y
    # NOTE: `var_discrete` takes indices (0-based) that should be treated as discrete.
    # We'll use grid=(2,2) per your example and apply identity func (no transform) for EPA.
    # If you want to use a link func (e.g., np.exp), pass func=np.exp.
    pdp_axes = pmb.plot_pdp(μ_bart, X=X, Y=Y, grid=(2, 2), func=None, var_discrete=[n_features-1])
    # pmb.plot_pdp returns matplotlib axes (or figure), save them
    # If it's axes (array), wrap in a figure
    try:
        fig = pdp_axes.figure
    except AttributeError:
        # If pdp_axes is array of axes, get the figure from the first axis
        fig = pdp_axes[0].figure

    # Adjust layout to prevent overlapping labels
    fig.tight_layout(pad=2.0)  # Add padding between subplots

    # Save the figure
    fpath = out_dir / "pmb_pdp.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info(f"Saved: {fpath}")

    # 4. Variable importance plot
    vi_results = pmb.compute_variable_importance(idata, μ_bart, X)
    fig = pmb.plot_variable_importance(vi_results, labels=feature_names)

    # Customize x-axis labels
    for ax in fig.axes:  # Iterate over all axes in the figure
        for label in ax.get_xticklabels():
            label.set_fontsize(8)  # Set smaller font size
            label.set_rotation(90)  # Rotate labels 90 degrees

    # Save the figure
    fpath = out_dir / "pmb_variable_importance.png"
    fig.figure.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig.figure)
    LOG.info(f"Saved variable importance plot to {fpath}")

    return idata, model, μ_bart, vi_results

embeddings_df = get_ball_flight_pct(embeddings_df.merge(
    tracking[['gpid','frame_id','pass_thrown']], on=['gpid','frame_id'], how='left'
))

embeddings_df = (
    embeddings_df.merge(
        plays[['gpid','ball_land_x','ball_land_y','down','yards_to_go','pass_length']], 
        on='gpid', 
        how='left'
    ).assign(
        down_x_dist=lambda x: x['down'] * x['yards_to_go'],
        pass_dist_to_goal=lambda x: np.where(
            x['ball_land_x'] >= 110,
            0,
            110 - x['ball_land_x']
        ),
        ball_land_dist_from_sideline=lambda x: np.where(
            x['ball_land_y'] <= 53.5 / 2,
            x['ball_land_y'],
            53.5 - x['ball_land_y']
        )
    ).drop(columns=['down','yards_to_go','ball_land_x','ball_land_y'])
)

gpids = embeddings_df['gpid'].unique()[:100]
embeddings_df = embeddings_df[embeddings_df['gpid'].isin(gpids)].reset_index(drop=True)

if __name__ == '__main__':
    # Prepare X, Y
    X, Y, feat_names = prepare_xy(embeddings_df)

    # Fit the model and save diagnostics
    idata, model, μ_bart, vi_results = fit_bart_model(X, Y, feature_names=feat_names, out_dir=OUT_DIR)

    # Save computed variable importance object (pickle)
    import pickle
    with open(OUT_DIR / "vi_results.pkl", "wb") as fh:
        pickle.dump(vi_results, fh)

    LOG.info("Done.")