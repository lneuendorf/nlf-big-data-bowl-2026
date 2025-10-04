import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
from IPython.display import HTML

def animate_play(tracking_play: pd.DataFrame, save_path: str = None):
    """Animate a single play from tracking data.

    Args:
        tracking_play: DataFrame containing tracking data for a single play (single gpid).
        save_path: Optional path to save the animation as a video file. If None, the 
        animation is displayed inline.

    Returns:
        If save_path is None, returns an HTML object containing the animation.
    """
    if tracking_play['gpid'].nunique() != 1:
        raise ValueError("DataFrame must contain only one unique play (gpid).")

    tracking_play = tracking_play.sort_values(by='frame_id')

    # --- Plot setup ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 53.3)
    ax.set_xlabel('Field Width (yards)')
    ax.set_ylabel('Yardline')
    ax.set_title(f"Play Animation: {tracking_play['gpid'].iloc[0]}")

    # Colors by side
    color_map = {'Offense': 'dodgerblue', 'Defense': 'orangered'}
    
    # Group by frame for animation
    frames = tracking_play.groupby('frame_id')
    
    # Set up scatter plot placeholders
    scat_off, = ax.plot([], [], 'o', color=color_map['Offense'], label='Offense', alpha=0.7)
    scat_def, = ax.plot([], [], 'o', color=color_map['Defense'], label='Defense', alpha=0.7)
    
    ax.legend(loc='upper right')
    

    # --- Y-axis follow logic ---
    window_height = 35  # yards
    y_bottom = max(0, min(frames.first()['y']) - 5)
    follow = False

    def get_y_limits(ball_y):
        nonlocal follow, y_bottom
        midpoint = y_bottom + window_height / 2
        if not follow:
            # Start moving only if ball crosses midpoint
            if ball_y > midpoint:
                follow = True
        if follow:
            y_bottom = ball_y - window_height / 2
        return y_bottom, y_bottom + window_height

    # --- Animation update ---
    def update(frame_tuple):
        frame_id, frame_data = frame_tuple

        y_min, y_max = get_y_limits(ball_y)

        ax.set_ylim(y_min, y_max)

        # Update scatter data
        off_data = frame_data[frame_data['player_side'] == 'Offense']
        def_data = frame_data[frame_data['player_side'] == 'Defense']

        scat_off.set_data(off_data['x'], off_data['y'])
        scat_def.set_data(def_data['x'], def_data['y'])
        
        ax.set_title(f"Frame {frame_id}")
        return scat_off, scat_def

    ani = animation.FuncAnimation(
        fig, update, frames=frames, blit=False, interval=80, repeat=False
    )

    if save_path:
        ani.save(save_path, writer='ffmpeg', fps=12)
        plt.close(fig)
        print(f"Animation saved to {save_path}")
    else:
        plt.close(fig)
        return HTML(ani.to_jshtml())

