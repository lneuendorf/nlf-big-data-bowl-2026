from typing import Dict
import textwrap
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as pe
import pandas as pd
import numpy as np
from IPython.display import HTML
from matplotlib.patches import FancyArrowPatch

def animate_play(
    tracking_play: pd.DataFrame, 
    play: pd.DataFrame,
    game: pd.DataFrame,
    team_desc: pd.DataFrame,
    save_path: str = None,
    plot_positions: bool = False,
    highlight_postpass_players:bool = False,
    show_postpass_paths: bool = False,
    plot_arrows: bool = False,
    show_ball_trajectory: bool = True,
    plot_heatmap: bool = False,
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
        highlight_postpass_players: If True, players involved in the pass after the ball
            is thrown will be highlighted.
        show_postpass_paths: If True, dashed lines will be drawn showing the paths of
            players involved in the pass after the ball is thrown.
        plot_arrows: If True, arrows will be drawn to indicate the direction of movement
            for players and the ball.

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

    # freeze and hold first frame for 10 frames
    first_frame = tracking_play['frame_id'].min()
    first_frame_data = tracking_play[tracking_play['frame_id'] == first_frame]
    for i in range(1, 11):
        temp = first_frame_data.copy()
        temp['frame_id'] = first_frame - i
        tracking_play = pd.concat([tracking_play, temp], axis=0)

    # freeze and hold last frame for 20 frames
    last_frame = tracking_play['frame_id'].max()
    last_frame_data = tracking_play[tracking_play['frame_id'] == last_frame]
    for i in range(1, 21):
        temp = last_frame_data.copy()
        temp['frame_id'] = last_frame + i
        tracking_play = pd.concat([tracking_play, temp], axis=0)

    # Initialize safety tracker and heatmap data
    safety_epa_line = None
    safety_current_epa = None
    
    if plot_heatmap:
        heatmap_data = (
            pd.read_parquet('../data/results/epa_predictions.parquet')
            .query('gpid == @tracking_play.gpid.values[0]')
            .loc[:, ['safety_nfl_id', 'num_frames_output', 'x', 'y', 'EPA']]
            .assign(
                x = lambda df: 120 - df['x'],
                y = lambda df: 53.3 - df['y']
            )
        ).head(4407)
        safety_id = heatmap_data['safety_nfl_id'].values[0]
        num_frames_output = heatmap_data['num_frames_output'].values[0]
        last_frame = tracking_play['frame_id'].max()
        heatmap_start_frame = max(0, last_frame - num_frames_output - 10 - 20) # one second before ball throw and 10 frame hold
        
        # Get EPA min/max for colorbar
        epa_min = heatmap_data['EPA'].min()
        epa_max = heatmap_data['EPA'].max()
        print(f"Heatmap EPA range: {epa_min:.2f} to {epa_max:.2f}")
        
        # Calculate nice tick positions
        epa_range = epa_max - epa_min
        if epa_range <= 0.1:
            tick_step = 0.02
        elif epa_range <= 0.5:
            tick_step = 0.1
        elif epa_range <= 1.0:
            tick_step = 0.2
        else:
            tick_step = 0.5
        
        # Generate ticks from min to max with step
        epa_ticks = np.arange(
            np.floor(epa_min / tick_step) * tick_step,
            np.ceil(epa_max / tick_step) * tick_step + tick_step/2,
            tick_step
        )
        epa_ticks = epa_ticks[(epa_ticks >= epa_min) & (epa_ticks <= epa_max)]
        
        # Ensure we have at least 3 ticks
        if len(epa_ticks) < 3:
            epa_ticks = np.linspace(epa_min, epa_max, 3)
    else:
        heatmap_data = None
        safety_id = None
        epa_min = 0
        epa_max = 0
        epa_ticks = []

    # Calculate x-axis limits based on play data
    all_x = tracking_play['x'].dropna()
    min_x = all_x.min()
    max_x = all_x.max()
    x_range = max_x - min_x
    
    # Ensure at least 50 yards of width, plus 2 yard buffer on each side
    if x_range < 50:
        center_x = (min_x + max_x) / 2
        min_x = center_x - 25 - 2
        max_x = center_x + 25 + 2
    else:
        min_x = min_x - 2
        max_x = max_x + 2

    min_x = 0 if min_x < 10 else min_x
    max_x = 120 if max_x > 110 else max_x
    
    # Ensure bounds are within reasonable field limits
    min_x = max(0, min_x)
    max_x = min(120, max_x)

    # --- Plot setup ---
    if plot_heatmap:
        # Create figure with adjusted width for colorbar
        fig, (ax, cax) = plt.subplots(
            1, 2, 
            figsize=(11.5, 7),
            gridspec_kw={'width_ratios': [10, 0.5], 'wspace': 0.05}
        )
    else:
        fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.set_xlim(min_x-.1, max_x+.1)
    ax.set_ylim(-0.1, 53.4)
    
    # Remove axis elements
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Remove the axis spine/border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
        
    # Add vertical yard lines every 5 yards on the x-axis
    for yard_line in range(max(int(np.ceil(min_x)), 10), min(int(np.floor(max_x)), 110) + 1):
        if yard_line % 5 == 0:
            ax.axvline(x=yard_line, color='gray', linestyle='-', alpha=0.5, zorder=1)

    # Add horizontal lines at top and bottom of the field
    ax.axhline(y=0, color='grey', linestyle='-', alpha=.5, zorder=1)
    ax.axhline(y=53.3, color='grey', linestyle='-', alpha=.5, zorder=1)

    # Add vertical lines at ends of field
    ax.axvline(x=0, color='grey', linestyle='-', alpha=.5, zorder=1)
    ax.axvline(x=120, color='grey', linestyle='-', alpha=.5, zorder=1)
    
    ax = _plot_yardline_numbers(ax, min_x, max_x)

    # NFL hash marks
    hash_width = .33
    hash_y_positions = [
        hash_width,
        ((53.3/2 * 3) - (18.5 / 2)) / 3 - hash_width, 
        ((53.3/2 * 3) + (18.5 / 2)) / 3 + hash_width,
        53.3 - hash_width
    ]
    for yard_line in range(max(int(np.ceil(min_x)), 10), min(int(np.floor(max_x)), 110) + 1):
        if yard_line % 5 != 0:
            for hash_y in hash_y_positions:
                ax.plot([yard_line, yard_line], [hash_y - hash_width, hash_y + hash_width], 
                    color='gray', linestyle='-', alpha=0.7, linewidth=1, zorder=1)
                
    # Plot the line of scrimmage
    los_x = play['absolute_yardline_number'].values[0]
    ax.axvline(x=los_x, color='blue', linestyle='-', alpha=0.8, linewidth=2, zorder=2)

    # Plot the first down line
    ytg = play['yards_to_go'].values[0]
    ax.axvline(x=los_x - ytg, color='yellow', linestyle='-', alpha=0.8, linewidth=2, zorder=2)

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
        'Offense': off_edge,
        'Defense': def_edge,
        'Ball': 'brown'
    }
    edge_color_map = {
        'Offense': off_main, 
        'Defense': def_main,
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
        bbox_to_anchor=(.93, 1.085),
    )

    home_colors = {
        'c1': offense_c1 if home_team_is_offense else defense_c1,
        'c2': offense_c2 if home_team_is_offense else defense_c1,
    }
    away_colors = {
        'c1': defense_c1 if home_team_is_offense else offense_c1,
        'c2': defense_c2 if home_team_is_offense else offense_c2,
    }

    ax = _add_game_info_text(ax, play, game, home_colors, away_colors)

    if min_x < 10:
        # plot the defense team endzone
        ax.add_patch(plt.Rectangle((0, 0), 10, 53.3, color='grey', alpha=0.1, zorder=0))
        ax.text(
            5, 53.3/2, team_desc.query('team_abbr == @defense').team_nick.values[0].upper(),
            fontsize=60, fontweight='bold', alpha=.5,
            rotation=90, ha='center', va='center', zorder=1,
            color=defense_c1,
            path_effects=[pe.withStroke(linewidth=7, foreground=defense_c2)]
        )
    if max_x > 110:
        # plot the offense team endzone
        ax.add_patch(plt.Rectangle((110, 0), 10, 53.3, color='grey', alpha=0.1, zorder=0))
        ax.text(
            115, 53.3/2, team_desc.query('team_abbr == @offense').team_nick.values[0].upper(),
            fontsize=60, fontweight='bold', alpha=.5,
            rotation=-90, ha='center', va='center', zorder=1,
            color=offense_c1, 
            path_effects=[pe.withStroke(linewidth=7, foreground=offense_c2)]
        )

    position_texts = []
    player_path_lines = {}
    arrows = []
    
    # Create heatmap scatter placeholder and colorbar
    if plot_heatmap:
        heatmap_scat = ax.scatter([], [], c=[], cmap='RdYlGn_r', vmin=epa_min, vmax=epa_max, 
                                 s=20, alpha=1, zorder=2)
        
        # Create colorbar with proper axis
        cbar = plt.colorbar(heatmap_scat, cax=cax, orientation='vertical')
        cbar.set_label('EPA', fontsize=15, fontweight='bold', rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=9)
        
        # Set custom ticks
        if len(epa_ticks) > 0:
            cbar.set_ticks(epa_ticks)
            # Format tick labels to show 2 decimal places
            tick_labels = [f'{tick:.2f}' for tick in epa_ticks]
            cbar.set_ticklabels(tick_labels)
        
        # Initialize safety EPA line on colorbar (initially invisible)
        safety_epa_line = cax.axhline(y=0, color='black', linewidth=2.5, linestyle='-', 
                                     alpha=0, zorder=10)  # Start with alpha=0 (invisible)
        
    else:
        heatmap_scat = None
        cax = None
        cbar = None
        safety_epa_line = None

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
        if not thrown_frames.empty and show_ball_trajectory:
            ball_path_line.set_data(thrown_frames['x'].values, thrown_frames['y'].values)
        else:
            ball_path_line.set_data([], [])

        # --- Highlight post-pass players ---
        if highlight_postpass_players:
            pass_has_been_thrown = any(tracking_play['frame_id'] <= current_frame_id)
            if pass_has_been_thrown and (tracking_play['pass_thrown'].any()):
                postpass_players = frame_data[frame_data['player_to_predict'] == True]
                if not postpass_players.empty:
                    off_colors = ['red' if pid in postpass_players['nfl_id'].values else 
                                  edge_color_map['Offense'] for pid in off_data['nfl_id'].values]
                    def_colors = ['red' if pid in postpass_players['nfl_id'].values else 
                                    edge_color_map['Defense'] for pid in def_data['nfl_id'].values]
                    scat_off.set_edgecolors(off_colors)
                    scat_off.set_linewidths([3 if c == 'red' else 2 for c in off_colors])
                    scat_def.set_edgecolors(def_colors)
                    scat_def.set_linewidths([3 if c == 'red' else 2 for c in def_colors])
        else:
            scat_off.set_edgecolors(edge_color_map['Offense'])
            scat_def.set_edgecolors(edge_color_map['Defense'])
            scat_off.set_linewidths(2)
            scat_def.set_linewidths(2)

        # --- Post-pass player paths ---
        if show_postpass_paths:
            predicted_ids = tracking_play.loc[tracking_play['player_to_predict'] == True, 'nfl_id'].unique()

            for pid in predicted_ids:
                player_frames = tracking_play[
                    (tracking_play['nfl_id'] == pid)
                    & (tracking_play['pass_thrown'] == True)
                    & (tracking_play['frame_id'] <= current_frame_id)
                ]

                if not player_frames.empty:
                    if pid not in player_path_lines:
                        line, = ax.plot([], [], '--', color='black', linewidth=2, alpha=0.8, zorder=2)
                        player_path_lines[pid] = line
                    player_path_lines[pid].set_data(player_frames['x'].values, player_frames['y'].values)
                elif pid in player_path_lines:
                    player_path_lines[pid].set_data([], [])
        else:
            for line in player_path_lines.values():
                line.set_data([], [])

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
        
        # --- Plot arrows for movement ---
        if plot_arrows:
            for arrow in arrows:
                arrow.remove()
            arrows.clear()

            for _, row in frame_data.iterrows():
                if row['position'] in ['QB','Ball']:
                    continue
                x, y = row['x'], row['y']
                length = 3.0 
                angle_rad = np.radians(row['dir'])

                dx = length * np.cos(angle_rad)
                dy = length * np.sin(angle_rad)

                arrow = FancyArrowPatch(
                    posA=(x, y),
                    posB=(x + dx, y + dy),
                    arrowstyle='-|>', color='grey', mutation_scale=15, linewidth=2, zorder=3
                )
                ax.add_patch(arrow)
                arrows.append(arrow)

        # Update heatmap and safety EPA tracking
        if plot_heatmap and heatmap_scat is not None:
            if frame_id >= heatmap_start_frame:
                # Update heatmap
                heatmap_scat.set_offsets(heatmap_data[['x', 'y']].values)
                heatmap_scat.set_array(heatmap_data['EPA'].values)

                # Find safety's current position
                safety_frame_data = frame_data[frame_data['nfl_id'] == safety_id]
                if not safety_frame_data.empty:
                    safety_x = safety_frame_data['x'].values[0]
                    safety_y = safety_frame_data['y'].values[0]
                    
                    # Find nearest EPA point to safety's position
                    distances = np.sqrt(
                        (heatmap_data['x'] - safety_x)**2 + 
                        (heatmap_data['y'] - safety_y)**2
                    )
                    nearest_idx = distances.idxmin()
                    safety_current_epa = heatmap_data.loc[nearest_idx, 'EPA']
                    
                    # Update safety EPA line on colorbar (make visible)
                    if safety_epa_line is not None:
                        safety_epa_line.set_ydata([safety_current_epa, safety_current_epa])
                        safety_epa_line.set_alpha(0.9)  # Make line visible
            else:
                # Heatmap not started yet
                heatmap_scat.set_offsets(np.empty((0, 2)))
                heatmap_scat.set_array(np.array([]))
                
                # Hide safety EPA line (keep it invisible)
                # if safety_epa_line is not None:
                    # safety_epa_line.set_alpha(0)  # Make line invisible

        # Collect all active artists
        animated_artists = [scat_off, scat_def, scat_ball, ball_path_line, *position_texts, *arrows]
        animated_artists.extend(player_path_lines.values())
        if heatmap_scat is not None:
            animated_artists.append(heatmap_scat)
        if safety_epa_line is not None:
            animated_artists.append(safety_epa_line)

        return animated_artists

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

    n_row_play_description = int(np.ceil(len(play_description) / 100))

    # === Custom title line built from separate text elements ===
    base_y = np.select(
        [n_row_play_description == 2, n_row_play_description >= 3],
        [1.12, 1.15], default=1.09
    )
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