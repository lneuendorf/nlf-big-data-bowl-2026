from typing import Dict, List
import pandas as pd
import numpy as np

#TODO: switch code to polars after finished

class PathPredictionDataset:
    def __init__(self):
        self.MAX_DEFENDERS = 4 # disculding the safety being modeled

    def process(self, df: pd.DataFrame) -> List:
        # Sort defender by those most likely to be involved in coverage
        df = df.sort_values(
            ['gpid', 'frame_id', 'player_role', 'within_10_yards_proba'],
            ascending=[True, True, True, False]
        ).reset_index(drop=True)

        plays = df['gpid'].unique()
        processed_plays = []
        safety_positions = {'S', 'FS', 'SS'}

        df = self._compute_velocity_components(df)

        for gpid in plays:
            play_df = df[df['gpid'] == gpid]
            safeties = (
                play_df
                  .query('player_role=="Defensive Coverage" and position.isin(@safety_positions)')
                  ['nfl_id'].values.tolist()
            )
            for safety_nfl_id in safeties:
                play_df = self._compute_relative_positions(play_df, safety_nfl_id)
                processed_play = self._process_play(play_df, safety_nfl_id)
                if processed_play is not None:
                    processed_plays.append(processed_play)

        return processed_plays
    
    def _compute_velocity_components(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(
            vx = df['s'] * np.cos(np.deg2rad(df['dir'])),
            vy = df['s'] * np.sin(np.deg2rad(df['dir']))
        )
    
    def _compute_relative_positions(self, play_df: pd.DataFrame, safety_nfl_id: int) -> pd.DataFrame:
        safety_df = play_df[play_df['nfl_id'] == safety_nfl_id][['frame_id', 'x', 'y', 'vx', 'vy']]
        safety_df = safety_df.rename(columns={
            'x': 'safety_x', 'y': 'safety_y', 
            'vx': 'safety_vx', 'vy': 'safety_vy'
        })

        play_df = play_df.merge(safety_df, on='frame_id', how='left')
        play_df['rel_x'] = play_df['x'] - play_df['safety_x']
        play_df['rel_y'] = play_df['y'] - play_df['safety_y']
        play_df['rel_vx'] = play_df['vx'] - play_df['safety_vx']
        play_df['rel_vy'] = play_df['vy'] - play_df['safety_vy']
        play_df['ball_land_rel_x'] = play_df['ball_land_x'] - play_df['safety_x']
        play_df['ball_land_rel_y'] = play_df['ball_land_y'] - play_df['safety_y']

        return play_df

    def _process_play(self, play_df: pd.DataFrame, safety_nfl_id: int) -> Dict:
        frames = play_df['frame_id'].unique()
        T = len(frames)

        safety_f = np.zeros((T, 4))  # x, y, vx, vy
        receiver_f = np.zeros((T, 4))
        defenders_f = np.zeros((T, self.MAX_DEFENDERS, 4))
        ball_f = np.zeros((T, 6))
        mask_f = np.zeros((T, self.MAX_DEFENDERS))
        target_f = np.zeros((T, 2))

        safety_cols = ['frame_id', 'x', 'y', 'vx', 'vy']
        safety_df = play_df[play_df['nfl_id'] == safety_nfl_id][safety_cols]

        relative_cols = ['frame_id', 'rel_x', 'rel_y', 'rel_vx', 'rel_vy']
        receiver_df = play_df[play_df['player_role'] == "Targeted Receiver"][relative_cols]
        defender_df = play_df.query('player_role=="Defensive Coverage" and nfl_id!=@safety_nfl_id')[relative_cols]
        ball_df = play_df[play_df['player_role'] == "Ball"][relative_cols]

        # TODO: add ball land x, y rel and the ball flight path pct, zone or man, abs speed?, weight, height, bmi
        # TODO: do you normalize the data?

        return None

class PathPredictionModel:
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # NOTE: split data on gpid to avoid data leakege when multiple safties on one play
        pass

    def predict(self, X: pd.DataFrame) -> pd.Series:
        # Implement prediction logic here
        return pd.Series(np.zeros(X.shape[0]))  # Placeholder implementation