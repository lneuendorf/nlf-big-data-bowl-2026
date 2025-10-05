import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
from IPython.display import HTML
import textwrap

def animate_play(
    tracking_play: pd.DataFrame, 
    play: pd.DataFrame,
    game: pd.DataFrame,
    save_path: str = None,
    plot_positions: bool = False
):
    """Animate a single play from tracking data.

    Args:
        tracking_play: DataFrame containing tracking data for a single play (single gpid).
        play: DataFrame containing the play level data for a single play.
        game: Optional DataFrame containing game level data for a single game.
        save_path: Optional path to save the animation as a video file. If None, the 
            animation is displayed inline.
        plot_positions: If True, player positions (e.g., WR, QB) will be annotated over 
            the players.

    Returns:
        If save_path is None, returns an HTML object containing the animation.
    """
    if tracking_play['gpid'].nunique() != 1:
        raise ValueError("Tracking DataFrame must contain only one unique play (gpid).")
    if play['gpid'].nunique() != 1:
        raise ValueError("Play DataFrame must contain only one unique play (gpid).")
    if game is not None and game['game_id'].nunique() != 1:
        raise ValueError("Game DataFrame must contain only one unique game (gpid).")

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
    # ax.set_aspect('equal') # Equal aspect ratio between x and y axes
    
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
                
    # Plot the line of scrimmage
    los_x = play['absolute_yardline_number'].values[0]
    ax.axvline(x=los_x, color='blue', linestyle='-', alpha=0.8, linewidth=2)

    # Plot the first down line
    ytg = play['yards_to_go'].values[0]
    ax.axvline(x=los_x + ytg, color='yellow', linestyle='-', alpha=0.8, linewidth=2)

    # Colors by side
    color_map = {
        'Offense': 'dodgerblue', 
        'Defense': 'orangered', 
        'Ball': 'brown'
    }
    
    # Group by frame for animation
    frames = tracking_play.groupby('frame_id')

    if plot_positions:
        marker_size=12
    else:
        marker_size=8
    
    # Set up scatter plot placeholders for all three groups
    scat_off, = ax.plot([], [], 'o', color=color_map['Offense'], label='Offense', alpha=0.7, markersize=marker_size, zorder=3)
    scat_def, = ax.plot([], [], 'o', color=color_map['Defense'], label='Defense', alpha=0.7, markersize=marker_size, zorder=3)
    scat_ball, = ax.plot([], [], 'o', color=color_map['Ball'], label='Ball', alpha=1.0, markersize=6, markeredgecolor='black', zorder=5)
    
    ax.legend(
        loc='center left',
        bbox_to_anchor=(.9, 1.085),
    )

    ax = _add_game_info_text(ax, play, game)

    position_texts = []
    def update(frame_tuple):
        frame_id, frame_data = frame_tuple

        off_data = frame_data[frame_data['player_side'] == 'Offense']
        def_data = frame_data[frame_data['player_side'] == 'Defense']
        ball_data = frame_data[frame_data['player_side'] == 'Ball']

        scat_off.set_data(off_data['x'], off_data['y'])
        scat_def.set_data(def_data['x'], def_data['y'])
        scat_ball.set_data(ball_data['x'], ball_data['y'])

        # Clear old texts
        for txt in position_texts:
            txt.remove()
        position_texts.clear()

        if plot_positions:
            for _, row in off_data.iterrows():
                t = ax.text(
                    row['x']+.05, row['y']-.05, row['position'],
                    color='black', fontsize=6, ha='center', va='center',
                    fontweight='bold', zorder=4
                )
                position_texts.append(t)
            for _, row in def_data.iterrows():
                t = ax.text(
                    row['x']+.05, row['y']-.05, row['position'],
                    color='black', fontsize=6, ha='center', va='center',
                    fontweight='bold', zorder=4
                )
                position_texts.append(t)

        # Return all artists so they’re redrawn each frame
        return [scat_off, scat_def, scat_ball, *position_texts]

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
    
def _plot_yardlines(ax, min_x, max_x) -> plt.Axes:
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

def _add_game_info_text(ax, play: pd.DataFrame, game: pd.DataFrame) -> plt.Axes:
    home_team = game['home_team_abbr'].values[0]
    away_team = game['visitor_team_abbr'].values[0]
    offense = play['possession_team'].values[0]
    offense_score = play['pre_snap_home_score'].values[0] \
        if play['possession_team'].values[0] == home_team \
        else play['pre_snap_visitor_score'].values[0]
    defense = play['defensive_team'].values[0]
    defense_score = play['pre_snap_visitor_score'].values[0] \
        if play['defensive_team'].values[0] == away_team \
        else play['pre_snap_home_score'].values[0]
    quarter = play['quarter'].values[0]
    game_clock = play['game_clock'].values[0]
    down = play['down'].values[0]
    distance = play['yards_to_go'].values[0]
    play_description = play['play_description'].values[0] +\
        f" Pass traveled {int(play['pass_distance'].values[0])} yards in the air in" +\
        f" {play['num_frames_output'].values[0] / 10:.1f} seconds."
    if play_description.startswith('('):
        play_description = play_description.split(')', 1)[1].strip()

    down_mapper = {1: '1st', 2: '2nd', 3: '3rd', 4: '4th'}
    play_info = (f"{away_team} @ {home_team} | "
                 f"{offense} {offense_score} - {defense} {defense_score} | "
                 f"Q{quarter} {game_clock} | "
                 f"{down_mapper[down]} & {distance}")

    ax.set_title(loc='left', y=1.12, label=play_info, fontsize=16, 
                 fontweight='bold', ha='left', va='top')

    wrapped_text = textwrap.fill(play_description, width=100)
    ax.text(
        0, 1.07, wrapped_text,
        ha='left',
        va='top',
        transform=ax.transAxes,
        fontsize=10,
        wrap=True
    )

    return ax