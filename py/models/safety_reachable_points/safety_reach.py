import numpy as np
from matplotlib.path import Path

def simulate_outer_points(
    init_pos,
    init_vel,
    bin_deg=5,
    dt_sim=0.1,
    sim_time=2.0,
    max_speed=7,
    max_acc=7.0,
    max_turn_rate=2.0,
    slow_turn_multiplier=2.5
):
    """
    Simulate maximum reachable positions in all directions for a player
    with physical constraints over a fixed time horizon.
    
    Returns:
        tuple: (outer_points, final_vels, angles)
            outer_points (ndarray): Shape (n_bins, 2). Maximum reachable positions
                                   in yards for each angular direction after sim_time.
            final_vels (ndarray): Shape (n_bins, 2). Final velocity vectors in yd/s
                                 at each outer point.
            angles (ndarray): Shape (n_bins,). Target angles in radians corresponding
                             to each angular bin (0 to 2π).
    """
    init_pos = np.array(init_pos, dtype=float)
    init_vel = np.array(init_vel, dtype=float)
    n_bins = int(360 // bin_deg)
    angles = np.linspace(0, 2*np.pi, n_bins, endpoint=False)
    
    outer_points = []
    final_vels = []
    all_positions = []
    all_velocities = []
    
    for theta in angles:
        pos = init_pos.copy()
        vel = init_vel.copy()
        speed = np.linalg.norm(vel)
        
        # Store position and velocity history for this angle
        pos_history = [pos.copy()]
        vel_history = [vel.copy()]
        
        t = 0.0
        while t < sim_time:
            # Target direction unit vector
            target_dir = np.array([np.cos(theta), np.sin(theta)])
            
            if speed < 1e-3:
                current_dir = target_dir.copy()
            else:
                current_dir = vel / speed
            
            # Project current velocity onto target direction
            vel_projection = np.dot(vel, target_dir)
            
            target_speed = max_speed
            
            # Calculate angular difference for turning
            cross = current_dir[0]*target_dir[1] - current_dir[1]*target_dir[0]
            dot = np.clip(np.dot(current_dir, target_dir), -1.0, 1.0)
            ang_diff = np.arctan2(cross, dot)
            
            # Adaptive turn rate based on speed
            if speed < 0.5:
                max_omega = max_turn_rate * slow_turn_multiplier
            else:
                speed_factor = max(speed, 0.5)
                max_omega = max_turn_rate / (1.0 + 0.2*speed_factor)
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
            current_speed_in_target_dir = np.dot(vel, target_dir)
            speed_diff = target_speed - current_speed_in_target_dir
            
            max_dv = max_acc * dt_sim
            
            if abs(speed_diff) > max_dv:
                delta_speed = np.sign(speed_diff) * max_dv
            else:
                delta_speed = speed_diff
            
            # Apply acceleration
            accel_dir = 0.7 * target_dir + 0.3 * new_dir
            accel_norm = np.linalg.norm(accel_dir)
            if accel_norm < 1e-6:
                accel_dir = target_dir
            else:
                accel_dir = accel_dir / accel_norm
            
            # Update velocity
            vel = vel + accel_dir * delta_speed
            
            # Enforce maximum speed magnitude
            speed = np.linalg.norm(vel)
            if speed > max_speed:
                vel = vel * (max_speed / speed)
            
            # Update position
            pos = pos + vel * dt_sim
            t += dt_sim
            
            # Store history
            pos_history.append(pos.copy())
            vel_history.append(vel.copy())
            
            # Early stop if we're aligned and at target speed
            if (abs(ang_diff) < 0.05 and 
                abs(target_speed - np.dot(vel, target_dir)) < 0.1):
                remaining = sim_time - t
                if remaining > 0:
                    pos = pos + vel * remaining
                    pos_history.append(pos.copy())
                    vel_history.append(vel.copy())
                break
        
        outer_points.append(pos.copy())
        final_vels.append(vel.copy())
        all_positions.append(np.array(pos_history))
        all_velocities.append(np.array(vel_history))
    
    return (
        np.array(outer_points), 
        np.array(final_vels), 
        angles
    )

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