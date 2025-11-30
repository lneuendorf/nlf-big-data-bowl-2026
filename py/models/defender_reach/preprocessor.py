""" Preprocessor for Defender Reach Model. """

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

SEED = 22
np.random.seed(SEED)

class DefenderReachDataset:
    """
    Builds a per-defender snapshot .1 seconds before the pass is thrown, and labels whether
    the defener ended within 10 yards of the ball landing point.
    """

    def __init__(self, prepass_seconds: float = 0.1, fps: int = 10):
        self.prepass_seconds = prepass_seconds
        self.fps = fps

    # ---- Public orchestrator ----
    def generate_defender_data(self, tracking: pd.DataFrame, plays: pd.DataFrame) -> pd.DataFrame:
        self._validate_inputs(tracking, plays)
        tracking_f, plays_f = self._select_valid_plays(tracking, plays)
        tracking_f, plays_f = self._drop_long_airtime_plays(tracking_f, plays_f, max_air_time_s=4.0)
        snap_frames = self._compute_snapshot_frame_before_pass(tracking_f)
        defender_df = self._build_defender_features(tracking_f, plays_f, snap_frames)
        return defender_df

    # ---- Private helpers ----
    def _validate_inputs(self, tracking: pd.DataFrame, plays: pd.DataFrame) -> None:
        required_tracking = {
            "gpid", "game_id", "nfl_id", "frame_id",
            "x", "y", "s", "dir", "position", "pass_thrown"
        }
        required_plays = {"gpid", "ball_land_x", "ball_land_y", "num_frames_output"}

        missing_t = required_tracking - set(tracking.columns)
        missing_p = required_plays - set(plays.columns)
        if missing_t:
            raise ValueError(f"tracking missing required columns: {missing_t}")
        if missing_p:
            raise ValueError(f"plays missing required columns: {missing_p}")

    def _defender_positions(self) -> set:
        return {'DE', 'OLB', 'CB', 'SS', 'FS', 'MLB', 'ILB', 'NT', 'DT', 'S', 'LB'}

    def _safety_positions(self) -> set:
        return {'SS', 'FS', 'S'}
    
    def _select_valid_plays(
        self, tracking: pd.DataFrame, plays: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Keep plays with a safety present after pass is thrown."""
        mask = tracking["position"].isin(self._safety_positions()) & tracking["pass_thrown"].astype(bool)
        valid_gpids = tracking.loc[mask, "gpid"].unique()
        tracking_f = tracking[tracking["gpid"].isin(valid_gpids)].copy()
        plays_f = plays[plays["gpid"].isin(valid_gpids)].copy()
        return tracking_f, plays_f

    def _drop_long_airtime_plays(
        self, tracking: pd.DataFrame, plays: pd.DataFrame, max_air_time_s: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Drop plays where ball is 'in the air' > max_air_time_s (assuming 10Hz)."""
        in_air_counts = (
            tracking.loc[tracking["pass_thrown"].astype(bool), ["gpid", "frame_id"]]
            .drop_duplicates(["gpid", "frame_id"])
            .groupby("gpid")
            .size()
            .div(self.fps)  # seconds
            .rename("time_ball_in_air")
            .reset_index()
        )
        drop_gpids = in_air_counts.loc[in_air_counts["time_ball_in_air"] > max_air_time_s, "gpid"].unique()
        LOG.info(f"Dropping {len(drop_gpids)} plays with >{max_air_time_s}s ball airtime")
        tracking_f = tracking[~tracking["gpid"].isin(drop_gpids)].copy()
        plays_f = plays[~plays["gpid"].isin(drop_gpids)].copy()
        return tracking_f, plays_f

    def _compute_snapshot_frame_before_pass(self, tracking: pd.DataFrame) -> pd.DataFrame:
        """
        For each play, find first frame with pass_thrown==True.
        Snapshot frame = first_pass_frame - prepass_seconds * fps (clipped to earliest available).
        """
        first_pass = (
            tracking.loc[tracking["pass_thrown"].astype(bool), ["gpid", "frame_id"]]
            .sort_values(["gpid", "frame_id"])
            .groupby("gpid", as_index=False)
            .first()
            .rename(columns={"frame_id": "first_pass_frame"})
        )
        offset = int(round(self.prepass_seconds * self.fps))
        first_pass["target_frame"] = first_pass["first_pass_frame"] - offset

        # If no frame <= target_frame, fallback to earliest frame in that play
        earliest = (
            tracking.groupby("gpid", as_index=False)["frame_id"]
            .min()
            .rename(columns={"frame_id": "earliest_frame"})
        )
        merge = first_pass.merge(earliest, on="gpid", how="left")
        merge["snapshot_frame"] = np.where(
            merge["target_frame"] >= merge["earliest_frame"],
            merge["target_frame"],
            merge["earliest_frame"],
        )
        return merge[["gpid", "snapshot_frame"]]

    def _build_defender_features(
        self,
        tracking: pd.DataFrame,
        plays: pd.DataFrame,
        snap_frames: pd.DataFrame
    ) -> pd.DataFrame:
        """Build features/labels at snapshot frame for defenders."""
        defenders = tracking.loc[tracking["position"].isin(self._defender_positions())].copy()
        defenders['gpid_nflid'] = defenders['gpid'].astype(str) + '_' + defenders['nfl_id'].astype(str)
        defenders_to_keep = defenders.query('pass_thrown==True')['gpid_nflid'].unique()
        defenders = defenders[defenders['gpid_nflid'].isin(defenders_to_keep)].copy().drop(columns=['gpid_nflid'])
        defenders = defenders.merge(snap_frames, on="gpid", how="left")
        defenders = defenders.loc[defenders["frame_id"] == defenders["snapshot_frame"]]

        defenders = defenders.merge(
            plays[["gpid", "ball_land_x", "ball_land_y", "num_frames_output"]],
            on="gpid", how="left"
        )

        defenders = defenders.assign(
            seconds_in_air=lambda d: d["num_frames_output"] / self.fps,
            dir_rad=lambda d: np.deg2rad(d["dir"]),
            dx=lambda d: d["ball_land_x"] - d["x"],
            dy=lambda d: d["ball_land_y"] - d["y"],
        )
        defenders = defenders.assign(
            dist_from_ball_land=lambda d: np.sqrt(d["dx"] ** 2 + d["dy"] ** 2),
            vx=lambda d: d["s"] * np.cos(d["dir_rad"]),
            vy=lambda d: d["s"] * np.sin(d["dir_rad"]),
        )
        eps = 1e-6
        defenders = defenders.assign(
            ux=lambda d: np.where(d["dist_from_ball_land"] > eps, d["dx"] / d["dist_from_ball_land"], 0.0),
            uy=lambda d: np.where(d["dist_from_ball_land"] > eps, d["dy"] / d["dist_from_ball_land"], 0.0),
        )
        defenders = defenders.assign(
            approach_rate=lambda d: d["vx"] * d["ux"] + d["vy"] * d["uy"],
            lateral_rate=lambda d: d["vx"] * (-d["uy"]) + d["vy"] * d["ux"],
            redirection_cost=lambda d: np.maximum(0.0, -d["approach_rate"]),
        )

        # Last known position for label
        last_pos = (
            tracking.sort_values(["gpid", "nfl_id", "frame_id"])
            .drop_duplicates(subset=["gpid", "nfl_id"], keep="last")
            .loc[:, ["gpid", "nfl_id", "x", "y"]]
            .rename(columns={"x": "x_last", "y": "y_last"})
        )
        defenders = defenders.merge(last_pos, on=["gpid", "nfl_id"], how="left")
        defenders = defenders.assign(
            last_dist_from_ball_land=lambda d: np.sqrt(
                (d["x_last"] - d["ball_land_x"]) ** 2 + (d["y_last"] - d["ball_land_y"]) ** 2
            ),
            within_10_yards=lambda d: (d["last_dist_from_ball_land"] <= 10.0).astype(int),
        )

        cols = [
            "gpid", "game_id", "nfl_id", "x", "y",
            "dist_from_ball_land", "approach_rate", "lateral_rate", "redirection_cost",
            "seconds_in_air", "last_dist_from_ball_land", "within_10_yards",
        ]
        cols = [c for c in cols if c in defenders.columns]
        out = defenders[cols].reset_index(drop=True)
        LOG.info(f"defender dataset built: {out.shape[0]} rows, {len(cols)} cols")
        return out