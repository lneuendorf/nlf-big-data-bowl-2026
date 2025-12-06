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
        - target_epa: float

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
        target_epa = s["target_epa"]

        # -----------------------------
        # 1. Assemble nodes
        # -----------------------------
        node_feats = []
        node_types = []

        # Receiver node
        node_feats.append([
            receiver["x"], receiver["y"],
            receiver["vx"], receiver["vy"]
        ])
        node_types.append(self.type_to_id["receiver"])

        # Ball node
        # If no vx,vy -> set to 0
        vx = ball.get("vx", 0.0)
        vy = ball.get("vy", 0.0)
        node_feats.append([ball["x"], ball["y"], vx, vy])
        node_types.append(self.type_to_id["ball"])

        # Defender nodes
        for d in defenders:
            node_feats.append([d["x"], d["y"], d["vx"], d["vy"]])
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

            edge_feats.append([dx, dy, dist, rel_speed_proj])

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
        y = torch.tensor([target_epa], dtype=torch.float)

        g = Data(
            x=node_feats,
            node_type=node_types,
            edge_index=edge_index,
            edge_attr=edge_attr,
            global_feats=global_feats,
            y=y
        )
        return g