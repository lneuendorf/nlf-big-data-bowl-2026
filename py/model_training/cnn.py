import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

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
    def __init__(self, in_channels, embedding_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_embed = nn.Linear(64, embedding_dim)
        self.fc_out = nn.Linear(embedding_dim, 1)  # predict EPA

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # (batch, 64, 1, 1)
        x = x.view(x.size(0), -1)
        embedding = self.fc_embed(x)
        output = self.fc_out(embedding).squeeze(-1)
        return output, embedding

# ---------- Training setup ----------
def train_cnn(frames_list, epa_list, embedding_dim=64, batch_size=16, lr=1e-3,
              epochs=50, patience=5):
    dataset = FrameDataset(frames_list, epa_list)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    in_channels = frames_list[0].shape[0]
    model = SpatialCNN(in_channels, embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds, _ = model(X_batch)
            loss = criterion(preds, y_batch.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)

        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - MSE: {epoch_loss:.4f}")

        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            no_improve = 0
            # save best model embeddings
            model.eval()
            with torch.no_grad():
                all_frames = torch.stack(frames_list)
                _, embeddings = model(all_frames)
                best_embeddings = embeddings.detach().numpy()
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered")
                break

    return model, best_embeddings