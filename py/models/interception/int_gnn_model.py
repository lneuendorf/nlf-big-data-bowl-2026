import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GATv2Conv
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score
)
import copy

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class IntGNN(nn.Module):
    def __init__(self, node_feat_dim=6, node_type_count=3,
                 edge_feat_dim=11, global_dim=6,
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

        # Output layer for binary classification
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


def get_class_weights(dataset):
    """Compute class weights based on training dataset imbalance."""
    labels = torch.cat([d.y for d in dataset])
    pos_weight = (len(labels) - labels.sum()) / labels.sum()
    return torch.tensor(pos_weight, dtype=torch.float32)


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

    model = IntGNN(
        node_feat_dim=train_dataset[0].x.shape[1],
        node_type_count=3,
        edge_feat_dim=train_dataset[0].edge_attr.shape[1],
        global_dim=train_dataset[0].global_feats.shape[-1],
        hidden=64
    )

    pos_weight = get_class_weights(train_dataset).to(torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

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
            loss = loss_fn(pred, batch.y.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs

        avg_train_loss = total_loss / len(train_loader.dataset)

        # --------------------
        # VALIDATION
        # --------------------
        if val_loader is None:
            LOG.info(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.3f}")
            continue  # no early stopping without a validation set

        model.eval()
        val_loss_total = 0

        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch)
                loss = loss_fn(pred, batch.y.float())
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


def evaluate_classification(model, dataset, batch_size=64, threshold=0.5):
    """Compute metrics suited for imbalanced binary classification."""
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()

    y_true_all = []
    y_pred_prob_all = []

    with torch.no_grad():
        for batch in loader:
            y_pred_logits = model(batch)
            y_pred_proba = torch.sigmoid(y_pred_logits)
            y_true = batch.y
            y_true_all.append(y_true.cpu())
            y_pred_prob_all.append(y_pred_proba.cpu())

    y_true_all = torch.cat(y_true_all).numpy()
    y_pred_prob_all = torch.cat(y_pred_prob_all).numpy()
    y_pred_all = (y_pred_prob_all >= threshold).astype(int)

    metrics = {
        "AUROC": roc_auc_score(y_true_all, y_pred_prob_all),
        "Average Precision": average_precision_score(y_true_all, y_pred_prob_all),
        "F1": f1_score(y_true_all, y_pred_all),
        "Precision": precision_score(y_true_all, y_pred_all),
        "Recall": recall_score(y_true_all, y_pred_all),
    }
    return metrics