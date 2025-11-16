""" Predicts if safety can reach within 5 yards of where the ball lands 1 second before 
the ball is thrown. """

import json
import logging
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

LOG = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("/Users/lukeneuendorf/projects/nfl-big-data-bowl-2026/data/models/safety_reach/")
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 22
np.random.seed(SEED)

class SafetyReachDataset:
    """
    Builds a per-safety snapshot .1 seconds before the pass is thrown, and labels whether
    the safety ended within 10 yards of the ball landing point.
    """

    def __init__(self, prepass_seconds: float = 0.1, fps: int = 10):
        self.prepass_seconds = prepass_seconds
        self.fps = fps

    # ---- Public orchestrator ----
    def generate_safety_data(self, tracking: pd.DataFrame, plays: pd.DataFrame) -> pd.DataFrame:
        self._validate_inputs(tracking, plays)
        tracking_f, plays_f = self._select_valid_plays(tracking, plays)
        tracking_f, plays_f = self._drop_long_airtime_plays(tracking_f, plays_f, max_air_time_s=4.0)
        snap_frames = self._compute_snapshot_frame_before_pass(tracking_f)
        safety_df = self._build_safety_features(tracking_f, plays_f, snap_frames)
        return safety_df

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

    def _safety_positions(self) -> set:
        return {"S", "FS", "SS"}

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

    def _build_safety_features(
        self,
        tracking: pd.DataFrame,
        plays: pd.DataFrame,
        snap_frames: pd.DataFrame
    ) -> pd.DataFrame:
        """Build features/labels at snapshot frame for safeties."""
        saf = tracking.loc[tracking["position"].isin(self._safety_positions())].copy()
        saf['gpid_nflid'] = saf['gpid'].astype(str) + '_' + saf['nfl_id'].astype(str)
        safties_to_keep = saf.query('pass_thrown==True')['gpid_nflid'].unique()
        saf = saf[saf['gpid_nflid'].isin(safties_to_keep)].copy().drop(columns=['gpid_nflid'])
        saf = saf.merge(snap_frames, on="gpid", how="left")
        saf = saf.loc[saf["frame_id"] == saf["snapshot_frame"]]

        saf = saf.merge(
            plays[["gpid", "ball_land_x", "ball_land_y", "num_frames_output"]],
            on="gpid", how="left"
        )

        saf = saf.assign(
            seconds_in_air=lambda d: d["num_frames_output"] / self.fps,
            dir_rad=lambda d: np.deg2rad(d["dir"]),
            dx=lambda d: d["ball_land_x"] - d["x"],
            dy=lambda d: d["ball_land_y"] - d["y"],
        )
        saf = saf.assign(
            dist_from_ball_land=lambda d: np.sqrt(d["dx"] ** 2 + d["dy"] ** 2),
            vx=lambda d: d["s"] * np.cos(d["dir_rad"]),
            vy=lambda d: d["s"] * np.sin(d["dir_rad"]),
        )
        eps = 1e-6
        saf = saf.assign(
            ux=lambda d: np.where(d["dist_from_ball_land"] > eps, d["dx"] / d["dist_from_ball_land"], 0.0),
            uy=lambda d: np.where(d["dist_from_ball_land"] > eps, d["dy"] / d["dist_from_ball_land"], 0.0),
        )
        saf = saf.assign(
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
        saf = saf.merge(last_pos, on=["gpid", "nfl_id"], how="left")
        saf = saf.assign(
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
        cols = [c for c in cols if c in saf.columns]
        out = saf[cols].reset_index(drop=True)
        LOG.info(f"Safety dataset built: {out.shape[0]} rows, {len(cols)} cols")
        return out


class SafetyReachModel:
    """
    XGBoost binary classifier with Optuna tuning.
    - train(..., save_model=True, save_dir=...) to fit and optionally save artifacts
    - load(...)/predict(...) to reuse a cached model
    """

    DEFAULT_FEATURES = [
        "dist_from_ball_land",
        "approach_rate",
        "lateral_rate",
        "redirection_cost",
        "seconds_in_air",
    ]
    MODEL_FILENAME = "safety_xgb.json"
    META_FILENAME = "safety_xgb_meta.json"

    def __init__(self, features: Optional[List[str]] = None, seed: int = SEED):
        self.features = features or self.DEFAULT_FEATURES
        self.seed = seed
        self.model: Optional[xgb.Booster] = None
        self.best_params: Optional[dict] = None
        self.best_iteration: Optional[int] = None
        self.best_threshold: float = 0.5

        # Quiet Optuna globally
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        np.random.seed(self.seed)

    # ------------------------------
    # Public API
    # ------------------------------
    def train(
        self,
        safety_df: pd.DataFrame,
        save_model: bool = True,
        save_dir: Path = DEFAULT_CACHE_DIR,
        n_trials: int = 50,
        n_splits: int = 5,
    ) -> dict:
        """Train with a single game-stratified split and Optuna hyperparam search."""
        X, y, groups = self._prepare_xy(safety_df)
        train_idx, valid_idx = next(
            StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.seed).split(X, y, groups)
        )

        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)

        def objective(trial):
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "booster": "gbtree",
                "seed": self.seed,
                "nthread": -1,
                # Regularization
                "lambda": trial.suggest_float("lambda", 1e-5, 10.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-5, 10.0, log=True),
                # Tree complexity
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 10.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
                # Learning & sampling
                "eta": trial.suggest_float("eta", 0.02, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            }
            model = xgb.train(
                params, dtrain,
                num_boost_round=1000,
                evals=[(dvalid, "valid")],
                early_stopping_rounds=10,
                verbose_eval=False,
            )
            preds = model.predict(dvalid)
            return log_loss(y_valid, preds)

        LOG.info("Starting Optuna hyperparameter search...")
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        self.best_params = {
            **study.best_trial.params,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "booster": "gbtree",
            "seed": self.seed,
            "nthread": -1,
        }
        LOG.info(f"Best params: {self.best_params}")

        # Final training
        self.model = xgb.train(
            self.best_params,
            dtrain,
            num_boost_round=2000,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=10,
            verbose_eval=False,
        )
        self.best_iteration = int(self.model.best_iteration)

        # Eval on train/valid
        train_preds = self.model.predict(dtrain)
        valid_preds = self.model.predict(dvalid)
        train_logloss = log_loss(y_train, train_preds)
        valid_logloss = log_loss(y_valid, valid_preds)
        train_auc = roc_auc_score(y_train, train_preds)
        valid_auc = roc_auc_score(y_valid, valid_preds)
        LOG.info(f"Best iteration: {self.best_iteration}")
        LOG.info(f"Train logloss={train_logloss:.4f}, AUC={train_auc:.4f}")
        LOG.info(f"Valid logloss={valid_logloss:.4f}, AUC={valid_auc:.4f}")

        # Best threshold by F1 on validation
        self.best_threshold, best_f1 = self._find_best_threshold(y_valid, valid_preds, f1_score)
        LOG.info(f"Optimal threshold={self.best_threshold:.4f}, best F1={best_f1:.4f}")

        # Save artifacts
        if save_model:
            save_dir.mkdir(parents=True, exist_ok=True)
            model_path = save_dir / self.MODEL_FILENAME
            meta_path = save_dir / self.META_FILENAME
            self.model.save_model(model_path.as_posix())
            meta = {
                "features": self.features,
                "seed": self.seed,
                "best_params": self.best_params,
                "best_iteration": self.best_iteration,
                "best_threshold": self.best_threshold,
            }
            meta_path.write_text(json.dumps(meta, indent=2))
            LOG.info(f"Saved model to {model_path}")
            LOG.info(f"Saved metadata to {meta_path}")

        return {
            "train_logloss": train_logloss,
            "valid_logloss": valid_logloss,
            "train_auc": train_auc,
            "valid_auc": valid_auc,
            "best_threshold": self.best_threshold,
            "best_iteration": self.best_iteration,
            "best_params": self.best_params,
        }

    def load(self, load_dir: Path = DEFAULT_CACHE_DIR) -> None:
        """Load model + metadata from cache directory."""
        model_path = load_dir / self.MODEL_FILENAME
        meta_path = load_dir / self.META_FILENAME
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Missing {model_path} or {meta_path}")

        self.model = xgb.Booster()
        self.model.load_model(model_path.as_posix())

        meta = json.loads(meta_path.read_text())
        self.features = meta.get("features", self.DEFAULT_FEATURES)
        self.seed = int(meta.get("seed", SEED))
        self.best_params = meta.get("best_params")
        self.best_iteration = int(meta.get("best_iteration", 0))
        self.best_threshold = float(meta.get("best_threshold", 0.5))
        LOG.info(f"Loaded model and metadata from {load_dir}")

    def predict(self, safety_df: pd.DataFrame, return_proba: bool = True) -> pd.DataFrame:
        """Add pred_proba and pred columns to a copy of safety_df."""
        if self.model is None:
            raise RuntimeError("Model not loaded/trained. Call load() or train() first.")

        self._check_features_exist(safety_df)
        dmat = xgb.DMatrix(safety_df[self.features])
        proba = self.model.predict(dmat)
        pred = (proba >= self.best_threshold).astype(int)

        out = safety_df.copy()
        out["pred_proba"] = proba
        out["pred"] = pred
        return out if return_proba else out[["pred"]]

    # ------------------------------
    # Internals
    # ------------------------------
    def _prepare_xy(self, safety_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        self._check_features_exist(safety_df)
        X = safety_df[self.features].copy()
        y = safety_df["within_10_yards"].values.astype(int)
        groups = safety_df["game_id"].values
        return X, y, groups

    def _check_features_exist(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

    @staticmethod
    def _find_best_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray, metric=f1_score, num_thresholds: int = 500):
        thresholds = np.linspace(0.0, 1.0, num_thresholds)
        preds_matrix = (y_pred_proba.reshape(-1, 1) >= thresholds.reshape(1, -1)).astype(int)
        scores = np.array([metric(y_true, preds_matrix[:, i]) for i in range(num_thresholds)])
        best_idx = scores.argmax()
        return thresholds[best_idx], scores[best_idx]