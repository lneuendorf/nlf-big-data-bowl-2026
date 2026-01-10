#!/usr/bin/env python
import logging
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from tqdm import tqdm
import nflreadpy as nfl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from matplotlib.path import Path

sys.path.append('../py')
from preprocess import preprocess

pd.set_option('display.max_columns', None)

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# N_WEEKS = 18  
# gpid = '2024010711_1919'
# safety_nfl_id = 52444
week = 15
gpid = '2023121602_115'
safety_nfl_id = 43387
FIELD_ALPHA = 0.5

game_id = int(gpid.split('_')[0])
play_id = int(gpid.split('_')[1])

##############  Load and preprocess the tracking + play data ##############
sup_data = pd.read_csv('../data/supplementary_data.csv')
tracking_input, tracking_output = pd.DataFrame(), pd.DataFrame()

tracking_input = (
    pd.concat([tracking_input, pd.read_csv(f'../data/train/input_2023_w{week:02d}.csv')], axis=0)
    .query("game_id == @game_id and play_id == @play_id")
)
tracking_output = (
    pd.concat([tracking_output, pd.read_csv(f'../data/train/output_2023_w{week:02d}.csv')], axis=0)
    .query("game_id == @game_id and play_id == @play_id")
)
games, plays, players, tracking = preprocess.process_data(tracking_input, tracking_output, sup_data)
one_sec_before_pass_frame_id = tracking.query('pass_thrown').frame_id.min() - 10
tracking = tracking[tracking.nfl_id.isin(tracking_output.nfl_id.unique()) | tracking.player_side.eq('Ball')]
tracking = tracking.query('frame_id >= @one_sec_before_pass_frame_id')
team_desc = preprocess.fetch_team_desc()
results = pd.read_parquet('../data/results/epa_predictions.parquet')
df = results.query("gpid == @gpid").copy()
tracking = tracking.query("gpid == @gpid").copy()
plays = plays.query("gpid == @gpid").copy()
tracking = tracking.assign(
    vx=lambda x: np.cos(np.deg2rad(x['dir'])) * x['s'],
    vy=lambda x: np.sin(np.deg2rad(x['dir'])) * x['s'],
)

# -----------------------------
# Safety + play identifiers
# -----------------------------
df = df.query("safety_nfl_id == @safety_nfl_id")

# -----------------------------
# Simulated points for heatmap
# -----------------------------
sim = df.query("sample_type == 'simulated'")
xs, ys = sim["x"].values, sim["y"].values

# -----------------------------
# Interpolation grid - with bounds
# -----------------------------
grid_res = 400
# Limit grid to field boundaries (0-53.3 for y-axis)
x_min, x_max = max(xs.min(), 0), min(xs.max(), 120)  # Assuming 120 for field length
y_min, y_max = max(ys.min(), 0), min(ys.max(), 53.3)

xi = np.linspace(x_min, x_max, grid_res)
yi = np.linspace(y_min, y_max, grid_res)
xi_grid, yi_grid = np.meshgrid(xi, yi)

# Convex hull mask
points = np.column_stack((xs, ys))
hull = ConvexHull(points)
hull_path = Path(points[hull.vertices])
mask = hull_path.contains_points(
    np.column_stack((xi_grid.ravel(), yi_grid.ravel()))
).reshape(xi_grid.shape)

# -----------------------------
# Heatmap configs
# -----------------------------
heatmaps = [
    ("EPA", "Overall EPA Heatmap", "RdYlGn_r"),
    ("EPA_INT", "Interception EPA Heatmap", "coolwarm"),
    ("EPA_COMP", "Completion EPA Heatmap", "PuOr"),
]

# Tracking Data
min_frame_id, max_frame_id = tracking.frame_id.min(), tracking.frame_id.max()
cols = ['frame_id', 'x', 'y', 'vx', 'vy']

safety = tracking.query("nfl_id == @safety_nfl_id").copy()[cols]
safety_start = safety.query("frame_id == @min_frame_id")
safety_end = safety.query("frame_id == @max_frame_id")
safety_path = safety.query("frame_id > @min_frame_id and frame_id < @max_frame_id")

ball = tracking.query("player_side == 'Ball' and pass_thrown").copy()[cols]
ball_min_frame_id = ball.frame_id.min()
ball_start = ball.query("frame_id == @ball_min_frame_id")
ball_end = ball.query("frame_id == @max_frame_id")
ball_path = ball.query("frame_id > @ball_min_frame_id and frame_id < @max_frame_id")

receiver = tracking.query("player_side == 'Offense'").copy()[cols]
receiver_start = receiver.query("frame_id == @min_frame_id")
receiver_end = receiver.query("frame_id == @max_frame_id")
receiver_path = receiver.query("frame_id > @min_frame_id and frame_id < @max_frame_id")

defenders = tracking.query("player_side == 'Defense' and nfl_id != @safety_nfl_id").copy()[['nfl_id'] + cols]
defenders_start = defenders.query("frame_id == @min_frame_id")
defenders_end = defenders.query("frame_id == @max_frame_id")
defenders_path = defenders.query("frame_id > @min_frame_id and frame_id < @max_frame_id")

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
                    ha='center', va='center', alpha=FIELD_ALPHA,
                    fontsize=14, fontweight='bold', color='grey', zorder=1)
            ax.text(x_pos, 53.3 - 12, str(yardline),
                    ha='center', va='center', alpha=FIELD_ALPHA,
                    fontsize=14, fontweight='bold', rotation=180, color='grey', zorder=1)
            continue

        # Determine triangle direction (left for left side, right for right side)
        is_left_side = x_pos < 60

        # --- Bottom numbers (upright) ---
        ax.text(x_pos, 12, str(yardline),
                ha='center', va='center', alpha=FIELD_ALPHA,
                fontsize=14, fontweight='bold', color='grey', zorder=1)
        triangle_y = 12.1
        if is_left_side:
            triangle_x = x_pos - triangle_offset
            triangle = plt.Polygon([[triangle_x, triangle_y],
                                    [triangle_x + triangle_length, triangle_y - triangle_height / 2],
                                    [triangle_x + triangle_length, triangle_y + triangle_height / 2]],
                                   closed=True, color='grey', alpha=FIELD_ALPHA, zorder=1)
        else:
            triangle_x = x_pos + triangle_offset
            triangle = plt.Polygon([[triangle_x, triangle_y],
                                    [triangle_x - triangle_length+.2, triangle_y - triangle_height / 2],
                                    [triangle_x - triangle_length+.2, triangle_y + triangle_height / 2]],
                                   closed=True, color='grey', alpha=FIELD_ALPHA, zorder=1)
        ax.add_patch(triangle)

        # --- Top numbers (upside down) ---
        ax.text(x_pos, 53.3 - 12, str(yardline),
                ha='center', va='center', alpha=FIELD_ALPHA,
                fontsize=14, fontweight='bold', rotation=180, color='grey', zorder=1)
        triangle_y = 53.3 - 12.2
        if is_left_side:
            triangle_x = x_pos - triangle_offset
            triangle = plt.Polygon([[triangle_x, triangle_y],
                                    [triangle_x + triangle_length, triangle_y - triangle_height / 2],
                                    [triangle_x + triangle_length, triangle_y + triangle_height / 2]],
                                   closed=True, color='grey', alpha=FIELD_ALPHA, zorder=1)
        else:
            triangle_x = x_pos + triangle_offset
            triangle = plt.Polygon([[triangle_x, triangle_y],
                                    [triangle_x - triangle_length+.2, triangle_y - triangle_height / 2],
                                    [triangle_x - triangle_length+.2, triangle_y + triangle_height / 2]],
                                   closed=True, color='grey', alpha=FIELD_ALPHA, zorder=1)
        ax.add_patch(triangle)

    # Add vertical yard lines every 5 yards on the x-axis
    for yard_line in range(max(int(np.ceil(min_x)), 10), min(int(np.floor(max_x)), 110) + 1):
            if yard_line % 5 == 0:  # Every 5 yards
                ax.axvline(x=yard_line, color='gray', linestyle='-', alpha=FIELD_ALPHA, zorder=1)

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
                    color='gray', linestyle='-', alpha=FIELD_ALPHA, linewidth=1, zorder=1)
                
    # Add horizontal lines at top and bottom of the field
    ax.axhline(y=0, color='grey', linestyle='-', alpha=FIELD_ALPHA, zorder=1)
    ax.axhline(y=53.3, color='grey', linestyle='-', alpha=FIELD_ALPHA, zorder=1)

    # Line of scrimmage
    los_x = plays['absolute_yardline_number'].values[0]
    ax.axvline(x=los_x, color='blue', linestyle='-', alpha=FIELD_ALPHA, linewidth=2, zorder=2)

    # Plot the first down line
    ytg = plays['yards_to_go'].values[0]
    ax.axvline(x=los_x + ytg, color='yellow', linestyle='-', alpha=.7, linewidth=2, zorder=2)

    return ax

# Create directory for saving images if it doesn't exist
output_dir = "output_plots"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Create and save separate plots
# -----------------------------
col_map = {
    "EPA": "Overall EPA",
    "EPA_INT": "Interception EPA",
    "EPA_COMP": "Completion EPA",
}

x_min = min(max(0, xs.min()), tracking.x.min())
x_max = max(min(120, xs.max()), tracking.x.max())

for i, (col, title, cmap) in enumerate(heatmaps):
    # Create individual figure for each plot
    fig, ax = plt.subplots(figsize=(4, 4))
    
    zi = griddata(
        (xs, ys),
        sim[col].values,
        (xi_grid, yi_grid),
        method="cubic",
        fill_value=np.nan,
    )
    zi[~mask] = np.nan
    from scipy.ndimage import gaussian_filter

    # zi = gaussian_filter(zi, sigma=30)

    # Plot yardline numbers and field markings
    ax = _plot_yardline_numbers(ax, x_min, x_max)
    
    # Apply field boundary mask
    field_mask = (yi_grid >= 0) & (yi_grid <= 53.3) & (xi_grid >= 0) & (xi_grid <= 120)
    zi[~field_mask] = np.nan

    im = ax.contourf(
        xi_grid, yi_grid, zi, levels=15, cmap=cmap, alpha=0.85, zorder=3
    )

    # Safety
    ax.quiver(
        safety_start.x, safety_start.y,
        safety_start.vx, safety_start.vy,
        angles='xy', scale_units='xy', scale=1,
        color='black', alpha=0.8, zorder=7
    )

    ax.scatter(
        safety_start.x, safety_start.y,
        color="black", s=60, alpha=0.8,
        zorder=7, label="Safety"
    )

    # Receiver
    ax.quiver(
        receiver_start.x, receiver_start.y,
        receiver_start.vx, receiver_start.vy,
        angles='xy', scale_units='xy', scale=1,
        color='red', alpha=0.8, zorder=6
    )

    ax.scatter(
        receiver_start.x, receiver_start.y,
        s=60, color="red", alpha=0.8,
        zorder=6, label="Receiver"
    )

    # Defenders
    # ax.quiver(
    #     defenders_start.x, defenders_start.y,
    #     defenders_start.vx, defenders_start.vy,
    #     angles='xy', scale_units='xy', scale=1,
    #     color='green', alpha=0.8, zorder=6
    # )
    # ax.scatter(
    #     defenders_start.x, defenders_start.y,
    #     s=60, color="green", alpha=0.8,
    #     zorder=6, label="Defender"
    # )

    # Ball
    ball_speed_scale =0.3
    ax.quiver(
        ball_start.x, ball_start.y,
        ball_path.vx.iloc[0] * ball_speed_scale, ball_path.vy.iloc[0] * ball_speed_scale,
        angles='xy', scale_units='xy', scale=1,
        color='brown', alpha=0.8, zorder=8
    )
    ax.scatter(
        ball_start.x, ball_start.y,
        s=40, color="brown", marker="D", edgecolor="black",
        label="Ball", zorder=8
    )
    ax.scatter(
        ball_end.x, ball_end.y,
        s=40, color="red", marker="x",
        zorder=8, label="Ball Landing Point"
    )

    # ax.set_title(title, fontsize=16)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.1, 53.4)

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

    cbar = fig.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label(col_map[col], fontsize=11)

    ax.legend(loc="lower left", bbox_to_anchor=(-0.1, 0.0))

    # Title
    ax.set_title("Pre-Throw Heatmap", fontsize=16)

    plt.tight_layout()
    
    # Save the figure as an image
    filename = f"{output_dir}/{title.replace(' ', '_').lower()}_prethrow.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    
    # Close the figure to free memory
    plt.close(fig)

for i, (col, title, cmap) in enumerate(heatmaps):
    # Create individual figure for each plot
    fig, ax = plt.subplots(figsize=(4, 4))
    
    zi = griddata(
        (xs, ys),
        sim[col].values,
        (xi_grid, yi_grid),
        method="cubic",
        fill_value=np.nan,
    )
    zi[~mask] = np.nan

    # Plot yardline numbers and field markings
    ax = _plot_yardline_numbers(ax, x_min, x_max)
    
    # Apply field boundary mask
    field_mask = (yi_grid >= 0) & (yi_grid <= 53.3) & (xi_grid >= 0) & (xi_grid <= 120)
    zi[~field_mask] = np.nan

    im = ax.contourf(
        xi_grid, yi_grid, zi, levels=40, cmap=cmap, alpha=0.85, zorder=3
    )

    # Safety
    ax.scatter(
        safety_start.x, safety_start.y,
        color="black", s=60, alpha=0.5,
        zorder=7, label="Safety Start"
    )
    ax.scatter(
        safety_end.x, safety_end.y,
        color="black", s=60,
        label="Safety End", zorder=7
    )
    ax.plot(
        safety_path.x, safety_path.y,
        linestyle=":", color="black", alpha=0.8,
        linewidth=2, zorder=6
    )

    # Receiver
    ax.scatter(
        receiver_start.x, receiver_start.y,
        s=60, color="red", alpha=0.4,
        zorder=6, label="Receiver Start"
    )
    ax.scatter(
        receiver_end.x, receiver_end.y,
        s=60, color="red", alpha=0.8,
        label="Receiver End", zorder=6
    )
    ax.plot(
        receiver_path.x, receiver_path.y,
        linestyle=":", color="red", alpha=0.5,
        linewidth=2, zorder=6
    )

    # Defenders
    # ax.scatter(
    #     defenders_start.x, defenders_start.y,
    #     s=60, color="green", alpha=0.4,
    #     zorder=6
    # )
    # ax.scatter(
    #     defenders_end.x, defenders_end.y,
    #     s=60, color="green", alpha=0.8,
    #     label="Defender", zorder=6
    # )
    # for nfl_id, group in defenders_path.groupby('nfl_id'):
    #     ax.plot(
    #         group.x, group.y,
    #         linestyle=":", color="green", alpha=0.5,
    #         linewidth=2, zorder=5
    #     )

    # Ball
    ax.scatter(
        ball_start.x, ball_start.y,
        s=40, color="brown", marker="D", edgecolor="black",
        label="Ball", zorder=8
    )
    ax.scatter(
        ball_end.x, ball_end.y,
        s=40, color="red", marker="x",
        zorder=8
    )
    ax.plot(
        ball_path.x, ball_path.y,
        linestyle="--", color="brown", alpha=0.6,
        linewidth=2, zorder=7
    )

    ax.set_title("Interception EPA", fontsize=16)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.1, 53.4)

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

    cbar = fig.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label(col_map[col], fontsize=11)
    
    # Add legend only to the first plot

    # ax.legend(loc="lower left", bbox_to_anchor=(-0.1, 0.0))

    plt.tight_layout()
    
    # Save the figure as an image
    filename = f"{output_dir}/{title.replace(' ', '_').lower()}_postthrow.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    
    # Close the figure to free memory
    plt.close(fig)

print("All plots have been saved to the 'output_plots' directory.")