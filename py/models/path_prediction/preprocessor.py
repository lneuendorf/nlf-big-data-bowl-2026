from typing import Dict, List, Optional
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOG = logging.getLogger(__name__)

class PathPredictionDataset:
    """
    Process tracking data for path prediction of defensive players (safeties).

    Returns per-play dicts with arrays:
      - safety:   (T, 4) [sx_raw, sy_norm, svx, svy]
      - receiver: (T, 4) [rx_rel, ry_rel, rvx_rel, rvy_rel]
      - ball:     (T, 7) [bx_rel, by_rel, bvx_rel, bvy_rel, ball_flight_pct, 
                          ball_land_rel_x, ball_land_rel_y]
      - defenders:(T, MAX_DEFENDERS, 4) [dx_rel, dy_rel, dvx_rel, dvy_rel]
      - mask:     (T, MAX_DEFENDERS)
      - globals:  (1,) [zone_coverage_flag]
      - target:   (T, 2) next safety (x,y)
    """
    FIELD_WIDTH = 53.3  # yards

    def __init__(self, max_defenders: int = 4):
        self.MAX_DEFENDERS = max_defenders

    # -------------------------
    # Public methods
    # -------------------------
    def process(self, df: pd.DataFrame) -> List[Dict]:
        
        LOG.info(f"Processing path prediction data for {df['gpid'].nunique()} plays")

        df = self._compute_velocity_components(df)

        # normalize y to [-1, 1]
        df["y_norm"] = (df["y"] / self.FIELD_WIDTH) * 2.0 - 1.0
        df["ball_land_y_norm"] = (df["ball_land_y"] / self.FIELD_WIDTH) * 2.0 - 1.0

        df = self._attach_ball_flight_pct(df)

        plays = df["gpid"].unique().tolist()
        processed_plays: List[Dict] = []

        safety_positions = {"S", "FS", "SS"}

        for gpid in tqdm(plays, desc="Processing plays"):
            play_df = df[df["gpid"] == gpid]

            safeties = (
                play_df[
                    (play_df["player_role"] == "Defensive Coverage")
                    & (play_df["position"].isin(safety_positions))
                ]["nfl_id"]
                .unique()
                .tolist()
            )

            if len(safeties) == 0:
                continue

            # sort defender ordering by those most likely to be within 10 yards
            defenders_overall = (
                play_df[play_df["player_role"] == "Defensive Coverage"]
                .groupby("nfl_id", observed=True)["within_10_yards_proba"]
                .max()
                .reset_index()
                .sort_values("within_10_yards_proba", ascending=False)
            )
            defenders_overall = defenders_overall["nfl_id"].tolist()

            for safety_nfl_id in safeties:
                play_df_ = self._attach_relative_features(play_df, safety_nfl_id)
                processed_play = self._process_play(play_df_, safety_nfl_id, defenders_overall)
                if processed_play is not None:
                    processed_plays.append(processed_play)

        return processed_plays

    # -------------------------
    # Private helpers
    # -------------------------
    def _compute_velocity_components(self, df: pd.DataFrame) -> pd.DataFrame:

        LOG.info(f"Computing velocity components for {df['gpid'].nunique()} plays")

        rad = np.deg2rad(pd.to_numeric(df["dir"], errors="coerce").values)
        s = pd.to_numeric(df["s"], errors="coerce").values
        df["vx"] = s * np.cos(rad)
        df["vy"] = s * np.sin(rad)
        return df

    def _attach_ball_flight_pct(self, df: pd.DataFrame) -> pd.DataFrame:
        bounds = (
            df.groupby("gpid", observed=True)["frame_id"].agg(["min", "max"])
              .rename(columns={"min": "min_frame", "max": "max_frame"})
        )
        df = df.merge(bounds, left_on="gpid", right_index=True, how="left")
        denom = df["max_frame"] - df["min_frame"]
        df["ball_flight_pct"] = 0.0
        valid = denom > 0
        df.loc[valid, "ball_flight_pct"] = \
            (df.loc[valid, "frame_id"] - df.loc[valid, "min_frame"]) / denom[valid]
        df = df.drop(columns=["min_frame", "max_frame"])
        return df

    def _attach_relative_features(self, df: pd.DataFrame, safety_nfl_id: int) -> pd.DataFrame:
        """
        Join safety reference (safety_x, safety_y_norm, safety_vx, safety_vy) per 
        (gpid, frame_id), then compute rel_x/rel_y/rel_vx/rel_vy for all rows.

        If multiple safeties exist for a frame, the merge will pick duplicates; we 
        assume one safety per role/frame.
        """

        LOG.info(f"Attaching relative features for safety nfl_id {safety_nfl_id} in {df['gpid'].nunique()} plays")

        saf = df[
            (df["nfl_id"] == safety_nfl_id)
        ].loc[:, ["gpid", "frame_id", "nfl_id", "x", "y_norm", "vx", "vy"]].copy()
        saf = saf.rename(columns={
            "x": "safety_x", 
            "y_norm": "safety_y_norm", 
            "vx": "safety_vx", 
            "vy": "safety_vy"
        })

        saf_cols = ["gpid", "frame_id", "safety_x", "safety_y_norm", "safety_vx", "safety_vy"]
        df = df.merge(
            saf.loc[:, saf_cols],
            on=["gpid", "frame_id"],
            how="left",
            validate="m:1")

        df["rel_x"] = df["x"] - df["safety_x"]
        df["rel_y"] = df["y_norm"] - df["safety_y_norm"]
        df["rel_vx"] = df["vx"] - df["safety_vx"]
        df["rel_vy"] = df["vy"] - df["safety_vy"]
        df["ball_land_rel_x"] = df["ball_land_x"] - df["safety_x"]
        df["ball_land_rel_y"] = df["ball_land_y_norm"] - df["safety_y_norm"]

        return df

    def _process_play(
        self,
        play_df: pd.DataFrame,
        safety_nfl_id: int,
        defenders_overall: Optional[List[int]] = None,
    ) -> Optional[Dict]:
        frames = sorted(play_df["frame_id"].unique().tolist())
        if len(frames) == 0:
            return None

        T = len(frames) - 1  # we predict next step, so T-1 targets

        # allocate arrays
        safety_f = np.zeros((T, 4), dtype=float)
        receiver_f = np.zeros((T, 4), dtype=float)
        defenders_f = np.zeros((T, self.MAX_DEFENDERS, 4), dtype=float)
        mask_f = np.zeros((T, self.MAX_DEFENDERS), dtype=float)
        ball_f = np.zeros((T, 7), dtype=float)
        globals_f = np.zeros((1,), dtype=float)
        target_f = np.zeros((T, 2), dtype=float)

        # globals: first zone_coverage value, default null -> 1 (zone)
        globals_f[0] = float(play_df["zone_coverage"].fillna(1).iloc[0])

        # safety rows
        safety_rows = play_df[play_df["nfl_id"] == safety_nfl_id].sort_values("frame_id")
        if safety_rows.shape[0] == 0:
            return None
        safety_np = safety_rows.loc[:, ["frame_id", "x", "y_norm", "vx", "vy"]].to_numpy()
        safety_lookup = {int(r[0]): r[1:].tolist() for r in safety_np}

        # receiver rows (Targeted Receiver)
        receiver_rows = (
            play_df[play_df["player_role"] == "Targeted Receiver"]
            .sort_values("frame_id")
        )
        if receiver_rows.shape[0] == 0:
            LOG.warning(f"No Targeted Receiver for gpid {play_df['gpid'].unique()[0]}")
        cols = ["frame_id", "rel_x", "rel_y", "rel_vx", "rel_vy"]
        receiver_np = receiver_rows.loc[:, cols].to_numpy()
        recv_lookup = {int(r[0]): r[1:].tolist() for r in receiver_np}

        # ball rows
        ball_rows = play_df[play_df["player_role"] == "Ball"].sort_values("frame_id")
        if ball_rows.shape[0] == 0:
            LOG.warning(f"No Ball data for gpid {play_df['gpid'].unique()[0]}")
        cols = ["frame_id", "rel_x", "rel_y", "rel_vx", "rel_vy", "ball_flight_pct", 
                "ball_land_rel_x", "ball_land_rel_y"]
        ball_np = ball_rows.loc[:, cols].to_numpy()
        ball_lookup = {int(r[0]): r[1:].tolist() for r in ball_np}

        # defenders subset excluding safety
        defenders_df = play_df[(play_df["player_role"] == "Defensive Coverage") & 
                               (play_df["nfl_id"] != safety_nfl_id)].copy()
        defenders_df = defenders_df.sort_values(["nfl_id", "frame_id"])

        defenders = defenders_df["nfl_id"].unique().tolist()
        present = [d for d in defenders_overall if d in defenders]
        chosen_def_ids = present[: self.MAX_DEFENDERS]

        defender_lookups = {} # Can be empty if safety is only nearby defender
        for did in chosen_def_ids:
            drows = defenders_df[defenders_df["nfl_id"] == did].sort_values("frame_id")
            if drows.shape[0] == 0:
                continue
            cols = ["frame_id", "rel_x", "rel_y", "rel_vx", "rel_vy"]
            dnp = drows.loc[:, cols].to_numpy()
            defender_lookups[did] = {int(r[0]): r[1:].tolist() for r in dnp}

        # Assign data to numpy arrays
        for i, frame in enumerate(frames[:-1]):  # last frame has no next
            # safety
            if frame in safety_lookup:
                safety_f[i] = np.array(safety_lookup[frame], dtype=np.float32)
            else:
                raise ValueError(
                    f"Missing safety data for gpid {play_df['gpid'].unique()[0]} frame {frame}"
                )

            # receiver
            if frame in recv_lookup:
                receiver_f[i] = np.array(recv_lookup[frame], dtype=np.float32)
            else:
                raise ValueError(
                    f"Missing receiver data for gpid {play_df['gpid'].unique()[0]} frame {frame}"
                )

            # ball
            if frame in ball_lookup:
                ball_f[i] = np.array(ball_lookup[frame], dtype=np.float32)
            else:
                raise ValueError(
                    f"Missing ball data for gpid {play_df['gpid'].unique()[0]} frame {frame}"
                )

            # defenders
            for slot_idx, did in enumerate(chosen_def_ids[: self.MAX_DEFENDERS]):
                lookup = defender_lookups.get(did, {})
                if frame in lookup:
                    defenders_f[i, slot_idx] = np.array(lookup[frame], dtype=np.float32)
                    mask_f[i, slot_idx] = 1.0
                else:
                    raise ValueError(
                        f"Missing defender {did} data for gpid {play_df['gpid'].unique()[0]} frame {frame}"
                    )

            # target (next x,y)
            next_frame = frames[i + 1]
            if next_frame in safety_lookup:
                next_safety_pos = safety_lookup[next_frame]
                target_f[i] = np.array(next_safety_pos[:2], dtype=np.float32)
            else:
                raise ValueError(
                    f"Missing next safety data for gpid {play_df['gpid'].unique()[0]} frame {next_frame}"
                )

        processed = {
            "gpid": int(play_df["gpid"].unique()[0]),
            "safety_nfl_id": int(safety_nfl_id),
            "frames": np.array(frames[:-1], dtype=int),
            "safety": safety_f,
            "receiver": receiver_f,
            "ball": ball_f,
            "defenders": defenders_f,
            "mask": mask_f,
            "globals": globals_f,
            "target": target_f,
        }

        return processed