import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
from IPython.display import HTML

def animate_play(
    tracking_play: pd.DataFrame, 
    play: pd.DataFrame,
    save_path: str = None
):
    """Animate a single play from tracking data.

    Args:
        tracking_play: DataFrame containing tracking data for a single play (single gpid).
        play: DataFrame containing the play level data for a single play.
        save_path: Optional path to save the animation as a video file. If None, the 
            animation is displayed inline.

    Returns:
        If save_path is None, returns an HTML object containing the animation.
    """
    if tracking_play['gpid'].nunique() != 1:
        raise ValueError("Tracking DataFrame must contain only one unique play (gpid).")
    if play['gpid'].nunique() != 1:
        raise ValueError("Play DataFrame must contain only one unique play (gpid).")

    tracking_play = tracking_play.sort_values(by='frame_id')

    # Calculate x-axis limits based on play data
    all_x = tracking_play['x'].dropna()
    min_x = all_x.min()
    max_x = all_x.max()
    x_range = max_x - min_x
    
    # Ensure at least 50 yards of width, plus 2 yard buffer on each side
    if x_range < 50:
        center_x = (min_x + max_x) / 2
        min_x = center_x - 25 - 2  # 25 yards on each side plus buffer
        max_x = center_x + 25 + 2
    else:
        min_x = min_x - 2  # Add 2 yard buffer
        max_x = max_x + 2  # Add 2 yard buffer
    
    # Ensure bounds are within reasonable field limits
    min_x = max(0, min_x)
    max_x = min(120, max_x)

    # --- Plot setup ---
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(-0.1, 53.4)  # Field width becomes y-axis
    ax.set_aspect('equal')
    
    # Remove axis elements
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_ylabel('')  # Remove y-axis label
    
    # Remove the axis spine/border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.set_title(f"Play Animation: {tracking_play['gpid'].iloc[0]}")
    
    # Add vertical yard lines every 5 yards on the x-axis
    for yard_line in range(int(np.ceil(min_x)), int(np.floor(max_x)) + 1):
        if yard_line % 5 == 0:  # Every 5 yards
            ax.axvline(x=yard_line, color='gray', linestyle='-', alpha=0.5)

    # Add horizontal lines at top and bottom of the field
    ax.axhline(y=0, color='grey', linestyle='-', alpha=.5)
    ax.axhline(y=53.3, color='grey', linestyle='-', alpha=.5)
    
    ax = _plot_yardlines(ax, min_x, max_x)

    # NFL hash marks (18'6" apart on inner edges)
    hash_width = .33
    hash_y_positions = [
        hash_width,
        ((53.3/2 * 3) - (18.5 / 2)) / 3 - hash_width, 
        ((53.3/2 * 3) + (18.5 / 2)) / 3 + hash_width,
        53.3 - hash_width
    ]
    for yard_line in range(int(np.ceil(min_x)), int(np.floor(max_x)) + 1):
        # Only draw hash marks for yards not divisible by 5
        if yard_line % 5 != 0:
            # Draw hash marks at both hash positions
            for hash_y in hash_y_positions:
                ax.plot([yard_line, yard_line], [hash_y - hash_width, hash_y + hash_width], 
                    color='gray', linestyle='-', alpha=0.7, linewidth=1)
    
    # Colors by side
    color_map = {
        'Offense': 'dodgerblue', 
        'Defense': 'orangered', 
        'Ball': 'brown'
    }
    
    # Group by frame for animation
    frames = tracking_play.groupby('frame_id')
    
    # Set up scatter plot placeholders for all three groups
    scat_off, = ax.plot([], [], 'o', color=color_map['Offense'], label='Offense', alpha=0.7, markersize=8)
    scat_def, = ax.plot([], [], 'o', color=color_map['Defense'], label='Defense', alpha=0.7, markersize=8)
    scat_ball, = ax.plot([], [], 'o', color=color_map['Ball'], label='Ball', alpha=1.0, markersize=6, markeredgecolor='black')
    
    ax.legend(
        loc='center right',
        bbox_to_anchor=(1.1, 0.9),
    )

    # --- Animation update ---
    def update(frame_tuple):
        frame_id, frame_data = frame_tuple

        # Update scatter data for all three groups
        off_data = frame_data[frame_data['player_side'] == 'Offense']
        def_data = frame_data[frame_data['player_side'] == 'Defense']
        ball_data = frame_data[frame_data['player_side'] == 'Ball']

        scat_off.set_data(off_data['x'], off_data['y'])
        scat_def.set_data(def_data['x'], def_data['y'])
        scat_ball.set_data(ball_data['x'], ball_data['y'])
        
        # Update title with frame info
        play_info = f"Play: {tracking_play['gpid'].iloc[0]} | Frame: {frame_id}"
        
        ax.set_title(play_info)
        
        return scat_off, scat_def, scat_ball

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=frames, blit=False, interval=80, repeat=False
    )

    if save_path:
        try:
            ani.save(save_path, writer='ffmpeg', fps=12, dpi=100)
            plt.close(fig)
            print(f"Animation saved to {save_path}")
        except Exception as e:
            print(f"Error saving animation: {e}")
            # Fallback to HTML display
            plt.close(fig)
            return HTML(ani.to_jshtml())
    else:
        plt.close(fig)
        return HTML(ani.to_jshtml())
    
def _plot_yardlines(ax, min_x, max_x):
    yardline_positions = [10, 20, 30, 40, 50, 40, 30, 20, 10]
    yardline_x_positions = [20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Triangles for each side
    triangle_offset = 2.2  # horizontal offset from text
    triangle_height = .3  # vertical triangle size
    triangle_length = .7

    for i, (yardline, x_pos) in enumerate(zip(yardline_positions, yardline_x_positions)):
        if not (min_x <= x_pos <= max_x):
            continue

        # Skip triangles around the 50
        if yardline == 50:
            ax.text(x_pos, 12, str(yardline),
                    ha='center', va='center',
                    fontsize=14, fontweight='bold', color='grey')
            ax.text(x_pos, 53.3 - 12, str(yardline),
                    ha='center', va='center',
                    fontsize=14, fontweight='bold', rotation=180, color='grey')
            continue

        # Determine triangle direction (left for left side, right for right side)
        is_left_side = x_pos < 60

        # --- Bottom numbers (upright) ---
        ax.text(x_pos, 12, str(yardline),
                ha='center', va='center',
                fontsize=14, fontweight='bold', color='grey')
        triangle_y = 12.1
        if is_left_side:
            triangle_x = x_pos - triangle_offset
            triangle = plt.Polygon([[triangle_x, triangle_y],
                                    [triangle_x + triangle_length, triangle_y - triangle_height / 2],
                                    [triangle_x + triangle_length, triangle_y + triangle_height / 2]],
                                   closed=True, color='grey')
        else:
            triangle_x = x_pos + triangle_offset
            triangle = plt.Polygon([[triangle_x, triangle_y],
                                    [triangle_x - triangle_length+.2, triangle_y - triangle_height / 2],
                                    [triangle_x - triangle_length+.2, triangle_y + triangle_height / 2]],
                                   closed=True, color='grey')
        ax.add_patch(triangle)

        # --- Top numbers (upside down) ---
        ax.text(x_pos, 53.3 - 12, str(yardline),
                ha='center', va='center',
                fontsize=14, fontweight='bold', rotation=180, color='grey')
        triangle_y = 53.3 - 12.2
        if is_left_side:
            triangle_x = x_pos - triangle_offset
            triangle = plt.Polygon([[triangle_x, triangle_y],
                                    [triangle_x + triangle_length, triangle_y - triangle_height / 2],
                                    [triangle_x + triangle_length, triangle_y + triangle_height / 2]],
                                   closed=True, color='grey')
        else:
            triangle_x = x_pos + triangle_offset
            triangle = plt.Polygon([[triangle_x, triangle_y],
                                    [triangle_x - triangle_length+.2, triangle_y - triangle_height / 2],
                                    [triangle_x - triangle_length+.2, triangle_y + triangle_height / 2]],
                                   closed=True, color='grey')
        ax.add_patch(triangle)

    return ax