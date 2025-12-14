from pygam import LinearGAM, s, f, te
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nflreadpy as nfl

SEED = 42
np.random.seed(SEED)

# -------------------------------
# 1. Load and prepare data
# -------------------------------
def load_and_prepare_data():
    seasons = range(2016, 2026)
    pbp = (
        nfl.load_pbp(seasons=list(seasons))
        .to_pandas()
        .assign(
            drive_points=lambda x: np.select(
                [
                    x.drive_end_transition == 'TOUCHDOWN',
                    x.drive_end_transition == 'FIELD_GOAL',
                ],
                [7, 3],
                default=0
            )
        )
        [['game_id','play_id','drive','yardline_100','half_seconds_remaining','down','ydstogo',
          'posteam_timeouts_remaining','defteam_timeouts_remaining','drive_points']]
        .dropna()
        .drop_duplicates(ignore_index=True)
    )
    return pbp

# -------------------------------
# 2. Train / validation / test split
# -------------------------------
df = load_and_prepare_data()
feature_cols = ['yardline_100','half_seconds_remaining','down','ydstogo',
        'posteam_timeouts_remaining','defteam_timeouts_remaining']
target_col = ['drive_points']

unique_games = df['game_id'].unique()
train_games, temp_games = train_test_split(unique_games, test_size=0.4, random_state=SEED)
val_games, test_games = train_test_split(temp_games, test_size=0.5, random_state=SEED)

X_train = df[df['game_id'].isin(train_games)][feature_cols]
y_train = df[df['game_id'].isin(train_games)][target_col]

X_val = df[df['game_id'].isin(val_games)][feature_cols]
y_val = df[df['game_id'].isin(val_games)][target_col]

X_test = df[df['game_id'].isin(test_games)][feature_cols]
y_test = df[df['game_id'].isin(test_games)][target_col]

print(f"Train samples: {len(y_train)}")
print(f"Val samples: {len(y_val)}")
print(f"Test samples: {len(y_test)}")

# -------------------------------
# 3. Fit GAM
# -------------------------------
# Smooth spline for continuous features, factor for categorical
gam = LinearGAM(
    s(0) +                # yardline_100 (continuous)
    s(1) +                # half_seconds_remaining (continuous)
    f(2) +                # down (categorical)
    s(3) +                # ydstogo (continuous)
    s(4) +                # posteam_timeouts_remaining
    s(5) +                # defteam_timeouts_remaining
    te(2, 3) +            # interaction: down x ydstogo
    te(1, 4)              # interaction: half_seconds_remaining x posteam_timeouts_remaining
)
gam.gridsearch(X_train.values, y_train.values)

# -------------------------------
# 4. Evaluate
# -------------------------------
def evaluate_model(model, X, y, name="Dataset"):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = root_mean_squared_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"{name} -> MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")
    return preds

train_preds = evaluate_model(gam, X_train, y_train, "Train")
val_preds = evaluate_model(gam, X_val, y_val, "Validation")
test_preds = evaluate_model(gam, X_test, y_test, "Test")

# -------------------------------
# 5. Plot predictions vs yardline
# -------------------------------
plt.figure(figsize=(10,6))
for down, ydstogo in [(1,10),(2,7),(3,4),(4,2)]:
    pred_y = []
    yardline_vals = range(1,101)
    for yl in yardline_vals:
        features = np.array([[yl, 900, down, ydstogo, 2, 2]])
        pred_y.append(gam.predict(features)[0])
    plt.plot(yardline_vals, pred_y, label=f'Down {down}, Yards to Go {ydstogo}')
plt.xlabel("Yardline (100 to Goal)")
plt.ylabel("Predicted Drive Points")
plt.title("GAM Predicted Drive Points")
plt.legend()
plt.grid(True)
plt.show()

# save model to ../../../data/models/
import joblib
joblib.dump(gam, '../../../data/models/gam_drive_points_model.pkl')