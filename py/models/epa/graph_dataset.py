from typing import List, Dict, Any
import torch
from torch_geometric.data import Data, Dataset
import math

class EPAGraphDataset(Dataset):
    """
    Each sample should provide:
        - receiver: dict with {x, y, vx, vy}
        - ball: dict with {x, y} (velocity optional)
        - defenders: list of dicts (variable size)
        - global_features: dict or tensor
        - target_epa: float, None at inference time

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
            rec_to_ball_dist
        ])
        node_types.append(self.type_to_id["receiver"])

        # Ball node
        ball_speed = math.sqrt(ball['vx']**2 + ball['vy']**2)
        node_feats.append([ball["x"], ball["y"], ball["vx"], ball["vy"], ball_speed, 0.0])  # ball distance to itself = 0
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
            xi, yi, vxi, vyi = node_feats[i]
            xj, yj, vxj, vyj = node_feats[j]

            dx = xj - xi
            dy = yj - yi
            dist = math.sqrt(dx*dx + dy*dy) + 1e-6

            # relative speed projected on direction i->j
            rvx = vxj - vxi
            rvy = vyj - vyi
            rel_speed_proj = (rvx * dx + rvy * dy) / dist
            
            # angular features
            angle = math.atan2(dy, dx)  # angle of edge
            vi_angle = math.atan2(vyi, vxi)  # node i velocity direction
            vj_angle = math.atan2(vyj, vxj)  # node j velocity direction
            angle_diff_i = angle - vi_angle  # is j in front of i's motion?
            angle_diff_j = vj_angle - angle  # is j moving toward/away from i?
            
            # Interaction type encoding
            type_i = node_types[i].item()
            type_j = node_types[j].item()
            
            edge_feats.append([
                dx, dy, dist, rel_speed_proj,
                math.cos(angle_diff_i), math.sin(angle_diff_i),
                math.cos(angle_diff_j), math.sin(angle_diff_j),
                type_i == 0 and type_j == 2,  # receiver-to-defender
                type_i == 2 and type_j == 0,  # defender-to-receiver
                type_i == 1,  # ball involved
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
        if "target_epa" in s and s["target_epa"] is not None:
            y = torch.tensor([s["target_epa"]], dtype=torch.float)
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