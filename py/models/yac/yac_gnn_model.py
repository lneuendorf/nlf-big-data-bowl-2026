import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GATv2Conv
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
import copy

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class YACGNN(nn.Module):
    def __init__(self, node_feat_dim=9, node_type_count=3,
                 edge_feat_dim=11, global_dim=10,
                 hidden=64):
        super().__init__()

        self.type_emb = nn.Embedding(node_type_count, 4)

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim + 4, hidden),
            nn.ReLU()
        )

        # Attention-based convolution
        self.conv1 = GATv2Conv(hidden, hidden, heads=4, concat=False, 
                               edge_dim=edge_feat_dim)

        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, hidden),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1)
        )

    def forward(self, data):
        x = data.x
        t = data.node_type
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        g = data.global_feats
        if g.dim() == 1:
            g = g.unsqueeze(0)

        type_embed = self.type_emb(t)
        x = torch.cat([x, type_embed], dim=1)

        h = self.node_encoder(x)
        h = F.relu(self.conv1(h, edge_index, edge_attr))

        pooled = global_mean_pool(h, data.batch)
        gproj = self.global_proj(g)

        combined = torch.cat([pooled, gproj], dim=1)
        out = self.head(combined)
        return out.squeeze(1)

def train_model(
    train_dataset,
    val_dataset=None,
    batch_size=32,
    lr=1e-3,
    epochs=20,
    patience=5,
):
    LOG.info("Starting model training")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model = YACGNN(
        node_feat_dim=train_dataset[0].x.shape[1],
        node_type_count=3,
        edge_feat_dim=train_dataset[0].edge_attr.shape[1],
        global_dim=train_dataset[0].global_feats.shape[-1],
        hidden=64
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):

        # --------------------
        # TRAINING
        # --------------------
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            pred = model(batch)
            loss = loss_fn(pred, batch.y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs

        avg_train_loss = total_loss / len(train_loader.dataset)

        # --------------------
        # VALIDATION
        # --------------------
        if val_loader is None:
            continue  # no early stopping without a validation set

        model.eval()
        val_loss_total = 0

        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch)
                loss = loss_fn(pred, batch.y)
                val_loss_total += loss.item() * batch.num_graphs

        avg_val_loss = val_loss_total / len(val_loader.dataset)
        LOG.info(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.3f} | Val Loss: {avg_val_loss:.3f}")

        # --------------------
        # EARLY STOPPING
        # --------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                LOG.info(f"Early stopping triggered at epoch {epoch}.")
                break

    # --------------------
    # RESTORE BEST MODEL
    # --------------------
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        LOG.info(f"Restored best model with val loss: {best_val_loss:.4f}")

    return model

def evaluate_split(model, dataset, batch_size=64):
    """Compute MAE, SmoothL1, and MSE for a dataset split."""
    loader = DataLoader(dataset, batch_size=batch_size)
    
    mae_loss = nn.L1Loss(reduction='sum')
    huber_loss = nn.SmoothL1Loss(reduction='sum')
    mse_loss = nn.MSELoss(reduction='sum')

    model.eval()

    mae_total = 0
    huber_total = 0
    mse_total = 0
    n = 0

    with torch.no_grad():
        for batch in loader:
            y_pred = model(batch)
            y_true = batch.y

            mae_total += mae_loss(y_pred, y_true).item()
            huber_total += huber_loss(y_pred, y_true).item()
            mse_total += mse_loss(y_pred, y_true).item()

            n += batch.num_graphs

    return {
        "MAE": mae_total / n,
        "SmoothL1": huber_total / n,
        "MSE": mse_total / n,
        "RMSE": np.sqrt(mse_total / n),
    }