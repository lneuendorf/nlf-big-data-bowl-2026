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

        # Receiver node
        rec_to_ball_dist = math.sqrt((receiver["x"] - ball["x"])**2 + (receiver["y"] - ball["y"])**2)
        rec_speed = math.sqrt(receiver["vx"]**2 + receiver["vy"]**2)
        node_feats.append([
            receiver["x"], receiver["y"],
            receiver["vx"], receiver["vy"],
            rec_speed,
            rec_to_ball_dist,
        ])
        node_types.append(self.type_to_id["receiver"])

        # Ball node
        ball_speed = math.sqrt(ball['vx']**2 + ball['vy']**2)
        node_feats.append([
            ball["x"], ball["y"], 
            ball["vx"], ball["vy"], 
            ball_speed, 
            0.0
        ])
        node_types.append(self.type_to_id["ball"])

        # Defender nodes
        for d in defenders:
            def_to_ball_dist = math.sqrt((d["x"] - ball["x"])**2 + (d["y"] - ball["y"])**2)
            def_speed = math.sqrt(d["vx"]**2 + d["vy"]**2)
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
            # Convert node features to Python floats
            xi, yi, vxi, vyi, _, _ = node_feats[i].tolist()
            xj, yj, vxj, vyj, _, _ = node_feats[j].tolist()

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

            # -----------------------------
            # Edge feature vector
            # -----------------------------
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