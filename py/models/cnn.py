import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# ---------- Path Config ----------
MODEL_PATH = '../data/models'
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
SAVE_PATH = os.path.join(MODEL_PATH, 'cnn_model_weights.pth')

def format_data_for_cnn_training(tracking):
    """
    For each play:
    - Find the top 3 nearest defenders to the ball landing location in the final frame.
    - Keep those defenders, the ball, and the receiver across all frames.
    - Pad missing defenders with dummy rows (x,y,s = 0).
    - Assign defender_rank = 1, 2, 3 by proximity.
    """

    # Distance to ball landing location
    tracking["dist_from_ball_land"] = np.sqrt(
        (tracking["x"] - tracking["ball_land_x"]) ** 2
        + (tracking["y"] - tracking["ball_land_y"]) ** 2
    )
    tracking["defender_rank"] = np.nan

    final_cols = [
        "gpid", "frame_id", "nfl_id", "pass_thrown",
        "player_role", "x", "y", "s", "dir", "defender_rank",
        "epa", "ball_land_x", "ball_land_y"
    ]

    result = []

    for gpid, play_df in tracking.groupby("gpid"):
        last_frame_id = play_df["frame_id"].max()

        # Get defenders in last frame sorted by proximity
        last_frame = play_df.query("frame_id == @last_frame_id")
        defenders_last = (
            last_frame.query("player_role == 'Defensive Coverage'")
            .sort_values("dist_from_ball_land")
            .head(3)
            .assign(defender_rank=lambda d: np.arange(1, len(d) + 1))
        )

        # Defender IDs and rank map
        rank_map = defenders_last.set_index("nfl_id")["defender_rank"].to_dict()
        defender_ids = list(rank_map.keys())
        num_defenders = len(defender_ids)

        # Keep only these defenders, plus ball + receiver
        play_keep = play_df[
            (play_df["player_role"].isin(["Ball", "Targeted Receiver"]))
            | (play_df["nfl_id"].isin(defender_ids))
        ][final_cols].copy()

        # Assign defender_rank where applicable
        play_keep["defender_rank"] = play_keep["nfl_id"].map(rank_map)

        # Add dummy defenders if fewer than 3
        if num_defenders < 3:
            dummy_rows = []
            for frame_id, frame_df in play_keep.groupby("frame_id"):
                for rank in range(num_defenders + 1, 4):
                    dummy_rows.append({
                        "gpid": gpid,
                        "frame_id": frame_id,
                        "nfl_id": -1,
                        "pass_thrown": frame_df["pass_thrown"].iloc[0],
                        "player_role": "Defensive Coverage",
                        "x": 0.0,
                        "y": 0.0,
                        "s": 0.0,
                        "dir": 0.0,
                        "defender_rank": rank,
                        "epa": frame_df["epa"].iloc[0],
                        "ball_land_x": frame_df["ball_land_x"].iloc[0],
                        "ball_land_y": frame_df["ball_land_y"].iloc[0]
                    })
            play_keep = pd.concat([play_keep, pd.DataFrame(dummy_rows)], ignore_index=True)
        result.append(play_keep)

    return pd.concat(result, ignore_index=True)

def make_spatial_grid(frame_df):

    entities = {
        "defenders": (
            frame_df.query("player_role == 'Defensive Coverage'")
            .sort_values("defender_rank")
            .head(3)
        ),
        "receiver": frame_df.query("player_role == 'Targeted Receiver'").head(1),
        "ball": frame_df.query("player_role == 'Ball'").head(1)
    }

    if entities["receiver"].empty or entities["ball"].empty:
        return None  # skip malformed frames

    receiver = entities["receiver"].iloc[0]
    ball = entities["ball"].iloc[0]

    # --- Compute defender features ---
    def_feats = []
    for _, def_row in entities["defenders"].iterrows():
        feats = {
            # 1. defender s_x, s_y
            "def_sx": def_row["s_x"],
            "def_sy": def_row["s_y"],

            # 2. defender x,y - receiver x,y
            "def_rel_rx": def_row["x"] - receiver["x"],
            "def_rel_ry": def_row["y"] - receiver["y"],

            # 3. defender s_x,y - receiver s_x,y
            "def_rel_rsx": def_row["s_x"] - receiver["s_x"],
            "def_rel_rsy": def_row["s_y"] - receiver["s_y"],

            # 4. defender x,y - ball_land x,y
            "def_rel_bx": def_row["x"] - ball["ball_land_x"],
            "def_rel_by": def_row["y"] - ball["ball_land_y"],

            # 5. defender s_x,y - ball_land s_x,y (we use ball s components)
            "def_rel_bsx": def_row["s_x"] - ball["s_x"],
            "def_rel_bsy": def_row["s_y"] - ball["s_y"],
        }
        def_feats.append(list(feats.values()))

    # Pad with zeros if < 3 defenders
    while len(def_feats) < 3:
        def_feats.append([0.0] * len(feats))

    # --- Receiver → Ball-land relations ---
    rec_feats = [
        receiver["x"] - ball["x"], # rx - bx
        receiver["y"] - ball["y"], # ry - by
        receiver["s_x"] - ball["s_x"], # rs_x - bs_x
        receiver["s_y"] - ball["s_y"],  # rs_y - bs_y
    ]

    # --- Ball features ---
    ball_feats = [
        ball["ball_land_x"] - ball["x"],  # ball x travel to landing
        ball["ball_land_y"] - ball["y"] # ball y travel to landing
    ]

    # --- Construct final tensor ---
    # shape = (8 rows, 3 columns, 2 channels)
    grid = np.zeros((8, 3, 2), dtype=np.float32)

    # Fill defender-related rows (rows 0–4, columns 0–2)
    for i in range(0,10,2):  # 5 feature types
        for j in range(3):  # 3 defenders
            grid[int(np.floor(i/2)), j, :] = def_feats[j][i], def_feats[j][i + 1]

    # Fill receiver features (rows 5 and 6)
    for j in range(3):
        grid[5, j, :] = rec_feats[0], rec_feats[1]  # rx - bx, ry - by
        grid[6, j, :] = rec_feats[2], rec_feats[3]  # rs_x - bs_x, rs_y - bs_y

    # Fill ball features (row 7)
    for j in range(3):
        grid[7, j, :] = ball_feats[0], ball_feats[1]  # ball x travel, ball y travel

    return grid

# ---------- Dataset wrapper ----------
class FrameDataset(Dataset):
    def __init__(self, frames_list, epa_list):
        self.frames = frames_list
        self.epa = epa_list

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx], self.epa[idx]

# ---------- CNN definition ----------
class SpatialCNN(nn.Module):
    def __init__(self, in_channels, embedding_dim=32, dropout_rate=0.25):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.pool = nn.AdaptiveAvgPool2d(1)  # squeezes (3×2) → (1×1)
        self.dropout = nn.Dropout(dropout_rate)

        self.fc_embed = nn.Linear(32, embedding_dim)
        self.fc_out = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # -> (batch, 32, 1, 1)
        x = x.view(x.size(0), -1)  # -> (batch, 32)
        x = self.dropout(x)
        embedding = self.fc_embed(x)
        output = self.fc_out(F.relu(embedding)).squeeze(-1)
        return output, embedding

# ---------- Training setup ----------
def train_cnn(
    gpid_list, frames_list, epa_list,
    embedding_dim=32, batch_size=16, lr=5e-4,
    epochs=200, patience=5, weight_decay=1e-4,
    dropout_rate=0.3, device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # --- Split data by gpid (80/10/10 train/val/test) ---
    unique_gpids = list(set(gpid_list))
    gpids_train, gpids_temp = train_test_split(unique_gpids, test_size=0.2, random_state=42)
    gpids_val, gpids_test = train_test_split(gpids_temp, test_size=0.5, random_state=42)

    def select_indices(gpids_subset):
        return [i for i, gpid in enumerate(gpid_list) if gpid in gpids_subset]

    train_indices = select_indices(gpids_train)
    val_indices = select_indices(gpids_val)
    test_indices = select_indices(gpids_test)

    def subset(lst, indices):
        return [lst[i] for i in indices]

    frames_train, epa_train = subset(frames_list, train_indices), subset(epa_list, train_indices)
    frames_val, epa_val = subset(frames_list, val_indices), subset(epa_list, val_indices)
    frames_test, epa_test = subset(frames_list, test_indices), subset(epa_list, test_indices)

    # --- Datasets & Loaders ---
    train_dataset = FrameDataset(frames_train, epa_train)
    val_dataset = FrameDataset(frames_val, epa_val)
    test_dataset = FrameDataset(frames_test, epa_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Model setup ---
    in_channels = frames_list[0].shape[0]
    model = SpatialCNN(in_channels, embedding_dim, dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    no_improve = 0
    best_embeddings = None

    # --- Training loop ---
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).float()
            optimizer.zero_grad()
            preds, _ = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        epoch_loss /= len(train_dataset)

        # ---- Validation ----
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device).float()
                preds, _ = model(X_val)
                val_loss += criterion(preds, y_val).item() * X_val.size(0)
        val_loss /= len(val_dataset)
        print(f"Epoch {epoch+1}/{epochs} - Train MSE: {epoch_loss:.4f} - Val MSE: {val_loss:.4f}")

        # ---- Early stopping on validation ----
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0

            # Save embeddings from validation set
            all_embeds = []
            with torch.no_grad():
                for X_val, _ in val_loader:
                    X_val = X_val.to(device)
                    _, emb = model(X_val)
                    all_embeds.append(emb.cpu().numpy())
            best_embeddings = np.concatenate(all_embeds, axis=0)
            best_model_state = model.state_dict()
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered")
                break

    # --- Load best model for testing ---
    model.load_state_dict(best_model_state)
    model.eval()

    # --- Predict on test set ---
    y_true_test, y_pred_test = [], []
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            preds, _ = model(X_test)
            y_true_test.extend(y_test.numpy())
            y_pred_test.extend(preds.cpu().numpy())

    # --- Compute overall test metrics ---
    test_mse = mean_squared_error(y_true_test, y_pred_test)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_true_test, y_pred_test)
    print(f"\nTest Results — MSE: {test_mse:.4f} | RMSE: {test_rmse:.4f} | R²: {test_r2:.4f}")

    # ----------------------------------------------------
    # --- Evaluate by % of play using all (train+val+test) data ---
    # ----------------------------------------------------
    print("\nComputing binned metrics across all plays...")

    # --- Combine all data ---
    all_frames = frames_list
    all_epa = epa_list
    all_gpids = gpid_list

    # --- Predict on all data ---
    all_true, all_pred, all_gpid = [], [], []
    with torch.no_grad():
        for i in range(0, len(all_frames), batch_size):
            X_batch = torch.stack(all_frames[i:i+batch_size]).to(device)
            y_batch = torch.tensor(all_epa[i:i+batch_size]).float()
            preds, _ = model(X_batch)
            all_true.extend(y_batch.numpy())
            all_pred.extend(preds.cpu().numpy())
            all_gpid.extend(all_gpids[i:i+batch_size])

    # --- Build DataFrame ---
    df_all = pd.DataFrame({
        "gpid": all_gpid,
        "y_true": all_true,
        "y_pred": all_pred
    })

    # Compute play progression (frame index normalized per gpid)
    df_all["frame_idx"] = df_all.groupby("gpid").cumcount()
    df_all["frame_count"] = df_all.groupby("gpid")["frame_idx"].transform("max") + 1
    df_all["pct_into_play"] = df_all["frame_idx"] / df_all["frame_count"]

    # Bin by 20% increments
    df_all["pct_bin"] = pd.cut(
        df_all["pct_into_play"],
        bins=np.linspace(0, 1, 6),
        labels=["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"],
        include_lowest=True
    )

    # --- Compute metrics for each bin ---
    print("\n--- All Data Metrics by % of Play ---")
    for bin_label, group in df_all.groupby("pct_bin"):
        if len(group) < 5:
            continue
        mse_bin = mean_squared_error(group["y_true"], group["y_pred"])
        rmse_bin = np.sqrt(mse_bin)
        r2_bin = r2_score(group["y_true"], group["y_pred"])
        print(f"{bin_label:>8} | n={len(group):5d} | MSE: {mse_bin:.4f} | RMSE: {rmse_bin:.4f} | R²: {r2_bin:.4f}")

    return model, best_embeddings

def save_cnn_model(model, save_path=SAVE_PATH):
    """
    Save the CNN model weights.
    """
    torch.save(model.state_dict(), save_path)

def load_cnn_model(in_channels, embedding_dim=32, load_path=SAVE_PATH):
    """
    Load the CNN model with the saved weights.
    """
    model = SpatialCNN(in_channels=in_channels, embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(load_path))
    model.eval()  # Set the model to evaluation mode
    return model