from typing import List, Dict, Any
import torch
from torch_geometric.data import Data, Dataset
import math

class IntGraphDataset(Dataset):
    """
    Each sample should provide:
        - absolute_yardline_number: int
        - receiver: dict with {x, y, vx, vy}
        - ball: dict with {x, y} (velocity optional)
        - defenders: list of dicts (variable size)
        - global_features: dict or tensor
        - target_int: float, None at inference time

    Preprocessing outside loader ensures consistent direction normalization.
    """

    def __init__(self, samples: List[Dict[str, Any]]):
        super().__init__()
        self.samples = samples

        # Node-type dictionary
        self.type_to_id = {"receiver": 0, "ball": 1, "defender": 2}

    def len(self):
        return len(self.samples)

    def get(self, idx: int) -> Data:
        s = self.samples[idx]

        absolute_yardline_number = s["absolute_yardline_number"]
        receiver = s["receiver"]
        ball = s["ball"]
        defenders = s["defenders"]  # list of dicts
        global_feats = s["global_features"]  # tensor-like

        # -----------------------------
        # 1. Assemble nodes
        # -----------------------------
        node_feats = []
        node_types = []

        # Helper to compute speed
        def speed(vx, vy):
            return math.sqrt(vx**2 + vy**2)

        # Receiver node
        rec_to_ball_dist = math.sqrt((receiver["x"] - ball["x"])**2 + (receiver["y"] - ball["y"])**2)
        rec_speed = speed(receiver["vx"], receiver["vy"])
        node_feats.append([
            receiver["x"], receiver["y"],
            receiver["vx"], receiver["vy"],
            rec_speed,
            rec_to_ball_dist
        ])
        node_types.append(self.type_to_id["receiver"])

        # Ball node
        ball_speed = speed(ball.get('vx', 0.0), ball.get('vy', 0.0))
        node_feats.append([
            ball["x"], ball["y"],
            ball.get("vx", 0.0), ball.get("vy", 0.0),
            ball_speed,
            0.0
        ])
        node_types.append(self.type_to_id["ball"])

        # Defender nodes
        for d in defenders:
            def_to_ball_dist = math.sqrt((d["x"] - ball["x"])**2 + (d["y"] - ball["y"])**2)
            def_speed = speed(d["vx"], d["vy"])
            node_feats.append([
                d["x"], d["y"],
                d["vx"], d["vy"],
                def_speed,
                def_to_ball_dist
            ])
            node_types.append(self.type_to_id["defender"])

        node_feats = torch.tensor(node_feats, dtype=torch.float)
        node_types = torch.tensor(node_types, dtype=torch.long)

        # -----------------------------
        # 1b. Compute "first on ball path" feature
        # -----------------------------
        # Ball landing point (if available)
        landing_x = ball.get("x_end", ball["x"])
        landing_y = ball.get("y_end", ball["y"])
        ball_start = torch.tensor([ball["x"], ball["y"]], dtype=torch.float)
        ball_end = torch.tensor([landing_x, landing_y], dtype=torch.float)
        ball_vec = ball_end - ball_start
        ball_dist = torch.norm(ball_vec) + 1e-6
        ball_dir = ball_vec / ball_dist

        node_positions = node_feats[:, :2]  # x, y positions
        relative_pos = node_positions - ball_start
        projection = (relative_pos @ ball_dir)  # scalar projection along trajectory
        perpendicular_dist = torch.norm(relative_pos - projection[:, None] * ball_dir, dim=1)

        threshold = 1.0  # adjust for field scale
        on_line = (perpendicular_dist < threshold) & (projection > 0) & (projection < ball_dist)

        is_first_on_path = torch.zeros(node_feats.size(0), dtype=torch.float)
        if on_line.any():
            # First player along the trajectory (closest to ball start)
            first_idx = torch.argmin(projection[on_line])
            idxs = torch.nonzero(on_line).squeeze()
            if idxs.ndim == 0:
                idxs = idxs.unsqueeze(0)
            is_first_on_path[idxs[first_idx]] = 1.0

        node_feats = torch.cat([node_feats, is_first_on_path.unsqueeze(1)], dim=1)

        # -----------------------------
        # 2. Build fully-connected edges
        # -----------------------------
        N = node_feats.size(0)
        src, dst = [], []
        for i in range(N):
            for j in range(N):
                if i != j:
                    src.append(i)
                    dst.append(j)

        edge_index = torch.tensor([src, dst], dtype=torch.long)

        # -----------------------------
        # 3. Edge features
        # -----------------------------
        edge_feats = []
        for i, j in zip(src, dst):
            xi, yi, vxi, vyi, _, _ = node_feats[i, :6].tolist()
            xj, yj, vxj, vyj, _, _ = node_feats[j, :6].tolist()

            type_i = node_types[i].item()
            type_j = node_types[j].item()

            dx = xj - xi
            dy = yj - yi
            dist = math.sqrt(dx * dx + dy * dy) + 1e-6

            rvx = vxj - vxi
            rvy = vyj - vyi
            rel_speed_proj = (rvx * dx + rvy * dy) / dist

            angle = math.atan2(dy, dx)
            vi_angle = math.atan2(vyi, vxi)
            vj_angle = math.atan2(vyj, vxj)

            angle_diff_i = angle - vi_angle
            angle_diff_j = vj_angle - angle

            cos_angle_i = math.cos(angle_diff_i)
            sin_angle_i = math.sin(angle_diff_i)
            cos_angle_j = math.cos(angle_diff_j)
            sin_angle_j = math.sin(angle_diff_j)

            edge_feats.append([
                dx,
                dy,
                dist,
                rel_speed_proj,
                cos_angle_i,
                sin_angle_i,
                cos_angle_j,
                sin_angle_j,
                type_i == self.type_to_id["receiver"] and type_j == self.type_to_id["defender"],
                type_i == self.type_to_id["defender"] and type_j == self.type_to_id["receiver"],
                type_i == self.type_to_id["ball"],
            ])

        edge_attr = torch.tensor(edge_feats, dtype=torch.float)

        # -----------------------------
        # 4. Global features
        # -----------------------------
        if isinstance(global_feats, dict):
            global_feats = torch.tensor([list(global_feats.values())], dtype=torch.float)
        else:
            global_feats = torch.tensor(global_feats, dtype=torch.float)

        # -----------------------------
        # 5. Create Graph Data object
        # -----------------------------
        if "target_int" in s and s["target_int"] is not None:
            y = torch.tensor([s["target_int"]], dtype=torch.float)
        else:
            y = None # at inference time

        g = Data(
            x=node_feats,
            node_type=node_types,
            edge_index=edge_index,
            edge_attr=edge_attr,
            global_feats=global_feats,
            y=y
        )
        return g
