from typing import Tuple
import pandas as pd
import numpy as np
import torch
from models.cnn import (
    make_spatial_grid,
    load_cnn_model
)

def generate_cnn_embeddings(frame_df: pd.DataFrame) -> Tuple[np.ndarray, float]:
    """ Generate CNN embeddings and EPA prediction for a given frame. """
    grid_tensor = make_spatial_grid(frame_df)
    grid_tensor = grid_tensor.unsqueeze(0)  # Add batch dimension (1, C, H, W)

    
    in_channels = grid_tensor.shape[1]
    model = load_cnn_model(in_channels=in_channels)

    with torch.no_grad():
        embedding, epa_prediction = model(grid_tensor)

    return embedding.squeeze(0).numpy(), epa_prediction.item()