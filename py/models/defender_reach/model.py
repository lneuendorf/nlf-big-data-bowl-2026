""" Predicts if defensive player can reach within 5 yards of where the ball lands 1 second 
before the ball is thrown. """

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

DEFAULT_CACHE_DIR = Path("/Users/lukeneuendorf/projects/nfl-big-data-bowl-2026/data/models/defender_reach/")
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 22
np.random.seed(SEED)

class DefenderReachModel:
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
    MODEL_FILENAME = "defender_xgb.json"
    META_FILENAME = "defender_xgb_meta.json"

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
        defender_df: pd.DataFrame,
        save_model: bool = True,
        save_dir: Path = DEFAULT_CACHE_DIR,
        n_trials: int = 50,
        n_splits: int = 5,
    ) -> dict:
        """Train with a single game-stratified split and Optuna hyperparam search."""
        X, y, groups = self._prepare_xy(defender_df)
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

    def predict(self, defender_df: pd.DataFrame) -> pd.DataFrame:
        """Add pred_proba and pred columns to a copy of defender_df.
        
        Args:
            defender_df: DataFrame with required feature columns and "game_id" column.

        Returns:
            DataFrame with "pred_proba" and "pred" (binary) columns added.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded/trained. Call load() or train() first.")

        self._check_features_exist(defender_df)
        dmat = xgb.DMatrix(defender_df[self.features])
        proba = self.model.predict(dmat)
        pred = (proba >= self.best_threshold).astype(int)

        out = defender_df.copy()
        out["pred_proba"] = proba
        out["pred"] = pred
        return out[["pred_proba", "pred"]].values
    
    def check_is_trained(self) -> bool:
        """Check if the model has been trained and saved"""
        model_path = DEFAULT_CACHE_DIR / self.MODEL_FILENAME
        meta_path = DEFAULT_CACHE_DIR / self.META_FILENAME
        model_exists = model_path.exists() and meta_path.exists()
        if model_exists:
            LOG.info("Trained model artifacts found.")
        else:
            LOG.info("No trained model artifacts found.")
        return model_exists

    # ------------------------------
    # Internals
    # ------------------------------
    def _prepare_xy(self, defender_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        self._check_features_exist(defender_df)
        X = defender_df[self.features].copy()
        y = defender_df["within_10_yards"].values.astype(int)
        groups = defender_df["game_id"].values
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