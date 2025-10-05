from typing import Dict
import textwrap
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as pe
import pandas as pd
import numpy as np
from IPython.display import HTML

def animate_play(
    tracking_play: pd.DataFrame, 
    play: pd.DataFrame,
    game: pd.DataFrame,
    team_desc: pd.DataFrame,
    save_path: str = None,
    plot_positions: bool = False
):
    """Animate a single play from tracking data.

    Args:
        tracking_play: DataFrame containing tracking data for a single play (single gpid).
        play: DataFrame containing the play level data for a single play.
        game: Optional DataFrame containing game level data for a single game.
        team_desc: DataFrame containing team descriptions (abbreviations, full names, logos, etc).
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
    for yard_line in range(max(int(np.ceil(min_x)), 10), min(int(np.floor(max_x)), 110) + 1):
        if yard_line % 5 == 0:  # Every 5 yards
            ax.axvline(x=yard_line, color='gray', linestyle='-', alpha=0.5, zorder=1)

    # Add horizontal lines at top and bottom of the field
    ax.axhline(y=0, color='grey', linestyle='-', alpha=.5, zorder=1)
    ax.axhline(y=53.3, color='grey', linestyle='-', alpha=.5, zorder=1)
    
    ax = _plot_yardline_numbers(ax, min_x, max_x)

    # NFL hash marks (18'6" apart on inner edges)
    hash_width = .33
    hash_y_positions = [
        hash_width,
        ((53.3/2 * 3) - (18.5 / 2)) / 3 - hash_width, 
        ((53.3/2 * 3) + (18.5 / 2)) / 3 + hash_width,
        53.3 - hash_width
    ]
    for yard_line in range(max(int(np.ceil(min_x)), 10), min(int(np.floor(max_x)), 110) + 1):
        # Only draw hash marks for yards not divisible by 5
        if yard_line % 5 != 0:
            # Draw hash marks at both hash positions
            for hash_y in hash_y_positions:
                ax.plot([yard_line, yard_line], [hash_y - hash_width, hash_y + hash_width], 
                    color='gray', linestyle='-', alpha=0.7, linewidth=1, zorder=1)
                
    # Plot the line of scrimmage
    los_x = play['absolute_yardline_number'].values[0]
    ax.axvline(x=los_x, color='blue', linestyle='-', alpha=0.8, linewidth=2, zorder=2)

    # Plot the first down line
    ytg = play['yards_to_go'].values[0]
    ax.axvline(x=los_x + ytg, color='yellow', linestyle='-', alpha=0.8, linewidth=2, zorder=2)

    # Colors by side
    def is_color_darker(color1, color2):
        color_brightness = {
            'black': 0, 'navy': 10, 'blue': 50, 'green': 50, 'red': 100,
            'purple': 60, 'gold': 200, 'yellow': 220, 'white': 255,
            'brown': 80, 'orange': 180, 'gray': 128
        }
        
        brightness1 = color_brightness.get(color1.lower(), 128)
        brightness2 = color_brightness.get(color2.lower(), 128)
        
        return brightness1 < brightness2

    offense = play.possession_team.values[0]
    home_team_is_offense = game.home_team_abbr.values[0] == offense
    offense_c1 = team_desc.query('team_abbr == @offense').team_color.values[0]
    offense_c2 = team_desc.query('team_abbr == @offense').team_color2.values[0]
    if is_color_darker(offense_c1, offense_c2):
        off_main, off_edge = offense_c1, offense_c2
    else:
        off_main, off_edge = offense_c2, offense_c1

    defense = play.defensive_team.values[0]
    defense_c1 = team_desc.query('team_abbr == @defense').team_color.values[0]
    defense_c2 = team_desc.query('team_abbr == @defense').team_color2.values[0]
    if is_color_darker(defense_c1, defense_c2):
        def_main, def_edge = defense_c1, defense_c2
    else:
        def_main, def_edge = defense_c2, defense_c1

    main_color_map = {
        'Offense': off_main,
        'Defense': def_main,
        'Ball': 'brown'
    }
    edge_color_map = {
        'Offense': off_edge, 
        'Defense': def_edge,
        'Ball': 'black'
    }
    
    # Group by frame for animation
    frames = tracking_play.groupby('frame_id')

    if plot_positions:
        marker_size=12
    else:
        marker_size=8
    
    # Set up scatter plot placeholders for all three groups
    scat_off = ax.scatter([], [], s=marker_size**2, c=main_color_map['Offense'],
                      edgecolors=edge_color_map['Offense'], linewidths=2,
                      label='Offense', alpha=0.7, zorder=3)
    scat_def = ax.scatter([], [], s=marker_size**2, c=main_color_map['Defense'],
                        edgecolors=edge_color_map['Defense'], linewidths=2,
                        label='Defense', alpha=0.7, zorder=3)
    scat_ball = ax.scatter([], [], s=6**2, c=main_color_map['Ball'],
                        edgecolors=edge_color_map['Ball'], linewidths=1,
                        label='Ball', alpha=1.0, zorder=5)
    
    # Dashed trajectory line for the ball (initially empty)
    (ball_path_line,) = ax.plot([], [], linestyle='--', color='brown', linewidth=2,
                            alpha=0.8, zorder=3)
    
    ax.legend(
        loc='center left',
        bbox_to_anchor=(.9, 1.085),
    )

    home_colors = {
        'c1': offense_c1 if home_team_is_offense else defense_c1,
        'c2': offense_c2 if home_team_is_offense else defense_c1,
    }
    away_colors = {
        'c1': defense_c1 if home_team_is_offense else offense_c1,
        'c2': defense_c1 if home_team_is_offense else offense_c2,
    }

    ax = _add_game_info_text(ax, play, game, home_colors, away_colors)

    position_texts = []
    def update(frame_tuple):
        frame_id, frame_data = frame_tuple

        off_data = frame_data[frame_data['player_side'] == 'Offense']
        def_data = frame_data[frame_data['player_side'] == 'Defense']
        ball_data = frame_data[frame_data['player_side'] == 'Ball']

        scat_off.set_offsets(off_data[['x', 'y']].values)
        scat_def.set_offsets(def_data[['x', 'y']].values)
        scat_ball.set_offsets(ball_data[['x', 'y']].values)

        # Plot dashed ball trajectory starting at pass_thrown
        current_frame_id = frame_id
        thrown_frames = tracking_play[
            (tracking_play['frame_id'] <= current_frame_id)
            & (tracking_play['pass_thrown'] == True)
            & (tracking_play['player_side'] == 'Ball')
        ]

        # Update the dashed trajectory line
        if not thrown_frames.empty:
            ball_path_line.set_data(thrown_frames['x'].values, thrown_frames['y'].values)
        else:
            ball_path_line.set_data([], [])

        # --- Positions text ---
        for txt in position_texts:
            txt.remove()
        position_texts.clear()
        if plot_positions:
            for _, row in off_data.iterrows():
                t = ax.text(
                    row['x'] + 0.05, row['y'] - 0.05, row['position'],
                    color='black', fontsize=6, ha='center', va='center',
                    fontweight='bold', zorder=4
                )
                position_texts.append(t)
            for _, row in def_data.iterrows():
                t = ax.text(
                    row['x'] + 0.05, row['y'] - 0.05, row['position'],
                    color='black', fontsize=6, ha='center', va='center',
                    fontweight='bold', zorder=4
                )
                position_texts.append(t)

        # Return all animated artists
        return [scat_off, scat_def, scat_ball, ball_path_line, *position_texts]

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
    
def _plot_yardline_numbers(ax, min_x, max_x) -> plt.Axes:
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
                    fontsize=14, fontweight='bold', color='grey', zorder=1)
            ax.text(x_pos, 53.3 - 12, str(yardline),
                    ha='center', va='center',
                    fontsize=14, fontweight='bold', rotation=180, color='grey', zorder=1)
            continue

        # Determine triangle direction (left for left side, right for right side)
        is_left_side = x_pos < 60

        # --- Bottom numbers (upright) ---
        ax.text(x_pos, 12, str(yardline),
                ha='center', va='center',
                fontsize=14, fontweight='bold', color='grey', zorder=1)
        triangle_y = 12.1
        if is_left_side:
            triangle_x = x_pos - triangle_offset
            triangle = plt.Polygon([[triangle_x, triangle_y],
                                    [triangle_x + triangle_length, triangle_y - triangle_height / 2],
                                    [triangle_x + triangle_length, triangle_y + triangle_height / 2]],
                                   closed=True, color='grey', zorder=1)
        else:
            triangle_x = x_pos + triangle_offset
            triangle = plt.Polygon([[triangle_x, triangle_y],
                                    [triangle_x - triangle_length+.2, triangle_y - triangle_height / 2],
                                    [triangle_x - triangle_length+.2, triangle_y + triangle_height / 2]],
                                   closed=True, color='grey', zorder=1)
        ax.add_patch(triangle)

        # --- Top numbers (upside down) ---
        ax.text(x_pos, 53.3 - 12, str(yardline),
                ha='center', va='center',
                fontsize=14, fontweight='bold', rotation=180, color='grey', zorder=1)
        triangle_y = 53.3 - 12.2
        if is_left_side:
            triangle_x = x_pos - triangle_offset
            triangle = plt.Polygon([[triangle_x, triangle_y],
                                    [triangle_x + triangle_length, triangle_y - triangle_height / 2],
                                    [triangle_x + triangle_length, triangle_y + triangle_height / 2]],
                                   closed=True, color='grey', zorder=1)
        else:
            triangle_x = x_pos + triangle_offset
            triangle = plt.Polygon([[triangle_x, triangle_y],
                                    [triangle_x - triangle_length+.2, triangle_y - triangle_height / 2],
                                    [triangle_x - triangle_length+.2, triangle_y + triangle_height / 2]],
                                   closed=True, color='grey', zorder=1)
        ax.add_patch(triangle)

    return ax

def _add_game_info_text(
    ax, 
    play: pd.DataFrame, 
    game: pd.DataFrame,
    home_colors: Dict,
    away_colors: Dict
) -> plt.Axes:
    # Extract basic info
    home_team = game['home_team_abbr'].values[0]
    away_team = game['visitor_team_abbr'].values[0]
    home_score = play['pre_snap_home_score'].values[0]
    away_score = play['pre_snap_visitor_score'].values[0]
    quarter = play['quarter'].values[0]
    game_clock = play['game_clock'].values[0]
    down = play['down'].values[0]
    distance = play['yards_to_go'].values[0]

    # Description text
    play_description = (
        play['play_description'].values[0]
        + f" Pass traveled {int(play['pass_distance'].values[0])} yards in the air"
        + f" in {play['num_frames_output'].values[0] / 10:.1f} seconds."
    )
    if play_description.startswith('('):
        play_description = play_description.split(')', 1)[1].strip()

    down_mapper = {1: '1st', 2: '2nd', 3: '3rd', 4: '4th'}

    # === Custom title line built from separate text elements ===
    base_y = 1.12
    x_cursor = 0.01
    font_size = 16

    # Away team abbreviation
    txt = ax.text(
        x_cursor, base_y, away_team,
        color=away_colors['c1'], fontsize=font_size, fontweight='bold',
        va='top', ha='left', transform=ax.transAxes,
        path_effects=[pe.withStroke(linewidth=2, foreground=away_colors['c2'])]
    )
    ax.figure.canvas.draw()  # needed to get text width
    renderer = ax.figure.canvas.get_renderer()
    text_bbox = txt.get_window_extent(renderer=renderer)
    # Convert bbox width from pixels to Axes coordinates
    x_cursor += (text_bbox.width / ax.figure.bbox.width) + .02

    # Away score + @
    text_segment = f" {away_score} @ "
    txt = ax.text(
        x_cursor, base_y, text_segment,
        color='black', fontsize=font_size, fontweight='bold',
        va='top', ha='left', transform=ax.transAxes,
    )
    ax.figure.canvas.draw()
    text_bbox = txt.get_window_extent(renderer=renderer)
    x_cursor += (text_bbox.width / ax.figure.bbox.width) + .02

    # Home team abbreviation (styled)
    txt = ax.text(
        x_cursor, base_y, home_team,
        color=home_colors['c1'], fontsize=font_size, fontweight='bold',
        va='top', ha='left', transform=ax.transAxes,
        path_effects=[pe.withStroke(linewidth=2, foreground=home_colors['c2'])]
    )
    ax.figure.canvas.draw()
    text_bbox = txt.get_window_extent(renderer=renderer)
    x_cursor += (text_bbox.width / ax.figure.bbox.width) + .02

    # Home score + rest of info
    text_segment = (f" {home_score} | Q{quarter} {game_clock} | "
                    f"{down_mapper[down]} & {distance}")
    ax.text(
        x_cursor, base_y, text_segment,
        color='black', fontsize=font_size, fontweight='bold',
        va='top', ha='left', transform=ax.transAxes
    )

    # Play description (wrapped)
    wrapped_text = textwrap.fill(play_description, width=100)
    ax.text(
        0.01, base_y - 0.05, wrapped_text,
        ha='left', va='top', transform=ax.transAxes,
        fontsize=10, wrap=True, color='black'
    )

    return ax