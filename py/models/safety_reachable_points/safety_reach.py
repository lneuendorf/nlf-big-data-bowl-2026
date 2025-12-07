import numpy as np
from matplotlib.path import Path

def simulate_outer_points(
    init_pos,
    init_vel,
    bin_deg=5,
    dt_sim=0.1,
    sim_time=2.0,
    max_speed=7,        # 7 yd/s ≈ 14.3 mph (slower than max nfl speed since this is short area)
    max_acc=7.0,          # 7.0 yd/s² ≈ 6.4 m/s²
    max_turn_rate=2.0,    #  2.0 rad/s ≈ 115°/s
    slow_turn_multiplier=2.5
):
    """
    Simulate maximum reachable positions in all directions for a player
    with physical constraints over a fixed time horizon.
    
    For each angular bin (around 360 degrees), simulate forward in small time-steps
    applying control that tries to reach and maintain speed in the target direction.
    Handles acceleration, deceleration, and turning to move in all directions.
    
    Args:
        init_pos (ndarray): Initial position [x, y] in yards
        init_vel (ndarray): Initial velocity [vx, vy] in yards/second
        bin_deg (float): Angular bin width in degrees. Smaller values give
                        higher angular resolution (default: 5°)
        dt_sim (float): Simulation time step in seconds (default: 0.1s = 10Hz)
        sim_time (float): Total simulation time horizon in seconds (default: 2.0s)
        max_speed (float): Maximum speed magnitude in yards/second (default: 7.0 yd/s)
        max_acc (float): Maximum acceleration magnitude in yards/second² (default: 12.0 yd/s²)
        max_turn_rate (float): Maximum turning rate in radians/second at moderate speed
                              (default: 3.0 rad/s ≈ 172°/s)
        slow_turn_multiplier (float): Multiplier for max_turn_rate when speed is very low
                                     (< 0.5 yd/s). Allows faster rotation in place
                                     (default: 2.0)
    
    Returns:
        tuple: (outer_points, final_vels, angles)
            outer_points (ndarray): Shape (n_bins, 2). Maximum reachable positions
                                   in yards for each angular direction after sim_time.
            final_vels (ndarray): Shape (n_bins, 2). Final velocity vectors in yd/s
                                 at each outer point.
            angles (ndarray): Shape (n_bins,). Target angles in radians corresponding
                             to each angular bin (0 to 2π).
    
    Note:
        - The simulation assumes a simple physics model with constant acceleration
          magnitude limits and speed-dependent turning capability.
        - At very low speeds (< 0.5 yd/s), turning capability is enhanced to model
          a player's ability to rotate in place.
        - The algorithm optimizes control to reach as far as possible in each
          direction within the time horizon, handling both acceleration and
          deceleration scenarios.
    """
    init_pos = np.array(init_pos, dtype=float)
    init_vel = np.array(init_vel, dtype=float)
    n_bins = int(360 // bin_deg)
    angles = np.linspace(0, 2*np.pi, n_bins, endpoint=False)
    
    outer_points = []
    final_vels = []
    
    for theta in angles:
        pos = init_pos.copy()
        vel = init_vel.copy()
        speed = np.linalg.norm(vel)
        
        t = 0.0
        while t < sim_time:
            # Target direction unit vector
            target_dir = np.array([np.cos(theta), np.sin(theta)])
            
            if speed < 1e-3:
                # If stationary, start moving directly in target direction
                current_dir = target_dir.copy()
            else:
                current_dir = vel / speed
            
            # Determine optimal target speed for this direction
            # Project current velocity onto target direction to see if we're moving
            # with or against it
            vel_projection = np.dot(vel, target_dir)
            
            if vel_projection > 0:
                # Moving somewhat in target direction - aim for max_speed
                target_speed = max_speed
            else:
                # Moving opposite to target direction - we need to decelerate and reverse
                # The target speed is still max_speed, but in the target direction
                target_speed = max_speed
            
            # Calculate angular difference for turning
            cross = current_dir[0]*target_dir[1] - current_dir[1]*target_dir[0]
            dot = np.clip(np.dot(current_dir, target_dir), -1.0, 1.0)
            ang_diff = np.arctan2(cross, dot)
            
            # Adaptive turn rate based on speed
            if speed < 0.5:
                max_omega = max_turn_rate * slow_turn_multiplier
            else:
                max_omega = max_turn_rate / (1.0 + 0.2*speed)
                max_omega = max(0.5, max_omega)
            
            # Apply turning within limits
            max_dtheta = max_omega * dt_sim
            dtheta = np.clip(ang_diff, -max_dtheta, max_dtheta)
            
            # Rotate current heading
            if speed < 1e-3:
                new_dir = target_dir if abs(ang_diff) < 1e-6 else np.array([
                    np.cos(dtheta), np.sin(dtheta)
                ])
            else:
                c, s = np.cos(dtheta), np.sin(dtheta)
                new_dir = np.array([c*current_dir[0] - s*current_dir[1],
                                    s*current_dir[0] + c*current_dir[1]])
            
            # Determine acceleration needed
            # We want to achieve target_speed in the target_dir direction
            # Current speed in target direction:
            current_speed_in_target_dir = np.dot(vel, target_dir)
            speed_diff = target_speed - current_speed_in_target_dir
            
            # Maximum speed change this timestep
            max_dv = max_acc * dt_sim
            
            if abs(speed_diff) > max_dv:
                delta_speed = np.sign(speed_diff) * max_dv
            else:
                delta_speed = speed_diff
            
            # Apply acceleration in the turning-adjusted direction
            # but bias toward target direction for more efficient movement
            accel_dir = 0.7 * target_dir + 0.3 * new_dir
            accel_dir = accel_dir / (np.linalg.norm(accel_dir) + 1e-6)
            
            # Update velocity
            vel = vel + accel_dir * delta_speed
            
            # Enforce maximum speed magnitude
            speed = np.linalg.norm(vel)
            if speed > max_speed:
                vel = vel * (max_speed / speed)
            
            # Update position
            pos = pos + vel * dt_sim
            t += dt_sim
            
            # Early stop if we're aligned and at target speed
            if (abs(ang_diff) < 0.05 and 
                abs(target_speed - np.dot(vel, target_dir)) < 0.1):
                remaining = sim_time - t
                if remaining > 0:
                    pos = pos + vel * remaining
                break
        
        outer_points.append(pos.copy())
        final_vels.append(vel.copy())
    
    return np.array(outer_points), np.array(final_vels), angles

def fill_polygon_with_grid(outer_pts, spacing=0.5):
    """
    Given ordered outer boundary points (should be roughly circular/order by angle),
    compute a bounding box grid with given spacing, then test points inside polygon using matplotlib.path.Path.
    Returns array of points inside.
    """
    # Ensure points are in order (by angle) for proper polygon creation
    center = outer_pts.mean(axis=0)
    angles = np.arctan2(outer_pts[:, 1] - center[1], outer_pts[:, 0] - center[0])
    sorted_idx = np.argsort(angles)
    ordered_pts = outer_pts[sorted_idx]
    
    path = Path(ordered_pts)
    minx, miny = ordered_pts.min(axis=0) - spacing
    maxx, maxy = ordered_pts.max(axis=0) + spacing
    xs = np.arange(minx, maxx+1e-6, spacing)
    ys = np.arange(miny, maxy+1e-6, spacing)
    gx, gy = np.meshgrid(xs, ys)
    grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    inside_mask = path.contains_points(grid_pts)
    inside = grid_pts[inside_mask]
    return inside, path, ordered_pts

def filter_points_near_defenders(points, defenders_last_positions, block_radius=0.5):
    """
    Remove points that are within block_radius of any defender last pos.
    """
    if len(points) == 0:
        return points
    if len(defenders_last_positions) == 0:
        return points
    
    d = np.linalg.norm(points[:, None, :] - defenders_last_positions[None, :, :], axis=2)
    mask = np.all(d >= block_radius, axis=1)
    return points[mask]