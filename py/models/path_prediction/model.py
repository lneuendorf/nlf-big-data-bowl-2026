from typing import Dict, List, Tuple
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOG = logging.getLogger(__name__)

MODELS_DIR = Path("/Users/lukeneuendorf/projects/nfl-big-data-bowl-2026/data/models/path_prediction/")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 2

# Global seeding for numpy / python / torch
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# Make CUDA deterministic (may slow training)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class PathPredictionDataset(Dataset):
    """Dataset class for path prediction with padding for variable-length sequences"""
    def __init__(self, processed_plays: List[Dict], max_sequence_length: int = None):
        self.processed_plays = processed_plays
        self.max_sequence_length = max_sequence_length or self._get_max_sequence_length()
        
    def _get_max_sequence_length(self) -> int:
        """Find the maximum sequence length in the dataset"""
        max_len = max(len(play['frames']) for play in self.processed_plays)
        LOG.info(f"Maximum sequence length in dataset: {max_len}")
        return max_len
        
    def __len__(self):
        return len(self.processed_plays)
    
    def __getitem__(self, idx):
        play = self.processed_plays[idx]
        seq_len = len(play['frames'])
        
        # Create sequence mask (1 for real data, 0 for padding)
        sequence_mask = torch.zeros(self.max_sequence_length)
        sequence_mask[:seq_len] = 1.0
        
        # Pad sequences to max_sequence_length
        def pad_tensor(tensor, target_length):
            current_length = tensor.shape[0]
            if current_length < target_length:
                # Calculate padding needed
                pad_length = target_length - current_length
                # Pad with zeros along the time dimension (dimension 0)
                padded = torch.cat([
                    tensor,
                    torch.zeros(pad_length, *tensor.shape[1:])
                ], dim=0)
                return padded
            elif current_length > target_length:
                # Truncate if longer (shouldn't happen with proper max_sequence_length)
                return tensor[:target_length]
            else:
                return tensor
        
        return {
            'safety': pad_tensor(torch.FloatTensor(play['safety']), self.max_sequence_length),
            'receiver': pad_tensor(torch.FloatTensor(play['receiver']), self.max_sequence_length),
            'ball': pad_tensor(torch.FloatTensor(play['ball']), self.max_sequence_length),
            'defenders': pad_tensor(torch.FloatTensor(play['defenders']), self.max_sequence_length),
            'defender_mask': pad_tensor(torch.FloatTensor(play['mask']), self.max_sequence_length),
            'globals': torch.FloatTensor(play['globals']),
            'target': pad_tensor(torch.FloatTensor(play['target']), self.max_sequence_length),
            'frames': pad_tensor(torch.LongTensor(play['frames']), self.max_sequence_length),
            'sequence_mask': sequence_mask,
        }

class SocialLSTMPathPrediction(nn.Module):
    """
    Social LSTM for safety path prediction with dynamic sequence and defender masking
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Regularization
        self.dropout = nn.Dropout(config.get('dropout_rate', 0.2))
        
        # Input encoders
        self.safety_encoder = self._build_mlp(
            input_dim=4,  # [sx_raw, sy_norm, svx, svy]
            hidden_dims=config['safety_encoder_dims'],
            output_dim=config['embedding_dim']
        )
        
        self.receiver_encoder = self._build_mlp(
            input_dim=4,  # [rx_rel, ry_rel, rvx_rel, rvy_rel]
            hidden_dims=config['receiver_encoder_dims'],
            output_dim=config['embedding_dim']
        )
        
        self.ball_encoder = self._build_mlp(
            input_dim=7,  # [bx_rel, by_rel, bvx_rel, bvy_rel, ball_flight_pct, ball_land_rel_x, ball_land_rel_y]
            hidden_dims=config['ball_encoder_dims'],
            output_dim=config['embedding_dim']
        )
        
        self.defender_encoder = self._build_mlp(
            input_dim=4,  # [dx_rel, dy_rel, dvx_rel, dvy_rel]
            hidden_dims=config['defender_encoder_dims'],
            output_dim=config['embedding_dim']
        )
        
        # Global feature encoder (zone coverage)
        self.global_encoder = nn.Linear(1, config['embedding_dim'])
        
        # Social pooling parameters
        self.pooling_conv = nn.Conv2d(
            in_channels=config['embedding_dim'],
            out_channels=config['social_pooling_dim'],
            kernel_size=3,
            padding=1
        )
            
        # LSTM with social context + individual defenders
        lstm_input_dim = (config['embedding_dim'] * (3 + config['max_defenders']) +  # safety, receiver, ball, each defender
                        config['social_pooling_dim'] +  # social context
                        config['embedding_dim'])  # global

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=config['lstm_hidden_dim'],
            num_layers=config['lstm_layers'],
            batch_first=True,
            dropout=config.get('lstm_dropout', 0.1) if config['lstm_layers'] > 1 else 0
        )
        
        # Output decoder
        self.output_decoder = self._build_mlp(
            input_dim=config['lstm_hidden_dim'],
            hidden_dims=config['decoder_dims'],
            output_dim=2  # next (x, y_norm)
        )
        
    def _build_mlp(self, input_dim: int, hidden_dims: List[int], output_dim: int) -> nn.Sequential:
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                self.dropout
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)
    
    def _social_pooling(self, defender_embeddings: torch.Tensor, 
                       defender_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply social pooling over defender embeddings
        
        Args:
            defender_embeddings: (batch_size, seq_len, max_defenders, embedding_dim)
            defender_mask: (batch_size, seq_len, max_defenders)
            
        Returns:
            social_context: (batch_size, seq_len, social_pooling_dim)
        """
        batch_size, seq_len, max_defenders, embedding_dim = defender_embeddings.shape
        
        # Apply mask to defender embeddings
        masked_embeddings = defender_embeddings * defender_mask.unsqueeze(-1)
        
        # Reshape for 2D convolution (treat spatial arrangement as 2D grid)
        # For simplicity, we'll use a 1D approach since we don't have explicit spatial grid
        # Sum over defenders to get aggregate social context
        social_context = torch.sum(masked_embeddings, dim=2)  # (batch_size, seq_len, embedding_dim)
        
        # Apply convolution for social pooling
        social_context = social_context.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        social_context = social_context.unsqueeze(-1)  # (batch_size, embedding_dim, seq_len, 1)
        
        social_context = self.pooling_conv(social_context)  # (batch_size, social_pooling_dim, seq_len, 1)
        social_context = social_context.squeeze(-1).transpose(1, 2)  # (batch_size, seq_len, social_pooling_dim)
        
        return social_context
    
    def forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        safety = batch['safety']
        receiver = batch['receiver']
        ball = batch['ball']
        defenders = batch['defenders']
        defender_mask = batch['defender_mask']
        globals_feat = batch['globals']
        sequence_mask = batch['sequence_mask']
        
        batch_size, seq_len, max_defenders = defenders.shape[:3]
        
        # Encode inputs
        safety_encoded = self.safety_encoder(safety)  # (batch_size, seq_len, embedding_dim)
        receiver_encoded = self.receiver_encoder(receiver)
        ball_encoded = self.ball_encoder(ball)
        
        # Encode defenders - keep individual embeddings
        defenders_flat = defenders.view(batch_size * seq_len * max_defenders, -1)
        defenders_encoded_flat = self.defender_encoder(defenders_flat)
        defenders_encoded = defenders_encoded_flat.view(
            batch_size, seq_len, max_defenders, self.config['embedding_dim']
        )
        
        # Apply defender mask and flatten defender dimension
        defenders_masked = defenders_encoded * defender_mask.unsqueeze(-1)
        defenders_combined = defenders_masked.view(batch_size, seq_len, -1)  # (batch_size, seq_len, max_defenders * embedding_dim)
        
        # Social pooling (optional - for additional social context)
        social_context = self._social_pooling(defenders_encoded, defender_mask)
        
        # Encode global features
        global_encoded = self.global_encoder(globals_feat.unsqueeze(-1))  # (batch_size, 1, embedding_dim)
        global_encoded = global_encoded.expand(-1, seq_len, -1)  # (batch_size, seq_len, embedding_dim)

        # Concatenate ALL features including individual defenders
        lstm_input = torch.cat([
            safety_encoded,           # (batch_size, seq_len, embedding_dim)
            receiver_encoded,         # (batch_size, seq_len, embedding_dim)
            ball_encoded,             # (batch_size, seq_len, embedding_dim)
            defenders_combined,       # (batch_size, seq_len, max_defenders * embedding_dim)
            social_context,           # (batch_size, seq_len, social_pooling_dim)
            global_encoded,           # (batch_size, seq_len, embedding_dim)
        ], dim=-1)
        
        # Apply sequence mask
        lstm_input = lstm_input * sequence_mask.unsqueeze(-1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(lstm_input)
        
        # Decode outputs
        predictions = self.output_decoder(lstm_out)
        
        return predictions, lstm_out

class PathPredictionTrainer:
    """Training and evaluation class with proper masking"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SocialLSTMPathPrediction(config).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
        self.criterion = nn.MSELoss(reduction='none')
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0  # Initialize patience counter
        
    def compute_masked_loss(self, predictions: torch.Tensor, target: torch.Tensor, 
                          sequence_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute loss only on non-padded positions
        """
        # Calculate loss per element
        loss_per_element = self.criterion(predictions, target)
        
        # Apply sequence mask (batch_size, seq_len) -> (batch_size, seq_len, 1)
        masked_loss = (loss_per_element * sequence_mask.unsqueeze(-1)).sum()
        
        # Count valid positions (each valid frame has 2 coordinates)
        valid_positions = sequence_mask.sum() * 2
        
        if valid_positions > 0:
            return masked_loss / valid_positions
        else:
            return masked_loss
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions, _ = self.model(batch)
            
            # Compute masked loss
            target = batch['target']
            sequence_mask = batch['sequence_mask']
            
            loss = self.compute_masked_loss(predictions, target, sequence_mask)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip', 1.0))
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches if num_batches > 0 else 0
    
    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                predictions, _ = self.model(batch)
                target = batch['target']
                sequence_mask = batch['sequence_mask']
                
                loss = self.compute_masked_loss(predictions, target, sequence_mask)
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches if num_batches > 0 else 0
    
    def test(self, test_loader: DataLoader) -> Dict:
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_masks = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                predictions, _ = self.model(batch)
                target = batch['target']
                sequence_mask = batch['sequence_mask']
                
                # Calculate loss
                loss = self.compute_masked_loss(predictions, target, sequence_mask)
                total_loss += loss.item()
                num_batches += 1
                
                # Store predictions and targets for additional metrics (only non-padded)
                batch_predictions = predictions.cpu().numpy()
                batch_targets = target.cpu().numpy()
                batch_masks = sequence_mask.cpu().numpy()
                
                # Only store non-padded elements
                for i in range(len(batch_predictions)):
                    valid_indices = batch_masks[i] == 1
                    if np.any(valid_indices):
                        all_predictions.append(batch_predictions[i][valid_indices])
                        all_targets.append(batch_targets[i][valid_indices])
        
        # Calculate additional metrics on non-padded data only
        if all_predictions:
            predictions_array = np.concatenate(all_predictions)
            targets_array = np.concatenate(all_targets)
            
            # Calculate RMSE
            mse_per_coord = np.mean((predictions_array - targets_array) ** 2, axis=0)
            rmse_per_coord = np.sqrt(mse_per_coord)
            overall_rmse = np.sqrt(np.mean((predictions_array - targets_array) ** 2))
            
            # Calculate MAE
            mae_per_coord = np.mean(np.abs(predictions_array - targets_array), axis=0)
            overall_mae = np.mean(np.abs(predictions_array - targets_array))
        else:
            rmse_per_coord = [0, 0]
            overall_rmse = 0
            mae_per_coord = [0, 0]
            overall_mae = 0
        
        metrics = {
            'average_loss': total_loss / num_batches if num_batches > 0 else 0,
            'rmse_x': rmse_per_coord[0],
            'rmse_y': rmse_per_coord[1],
            'overall_rmse': overall_rmse,
            'mae_x': mae_per_coord[0],
            'mae_y': mae_per_coord[1],
            'overall_mae': overall_mae
        }
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              test_loader: DataLoader, epochs: int, patience: int = 10):

        for epoch in range(epochs):

            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validation
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            LOG.info(f"Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Early stopping and model saving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_path = MODELS_DIR / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'config': self.config
                }, save_path.as_posix())
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= patience:
                LOG.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best model and test
        checkpoint_path = MODELS_DIR / 'best_model.pth'
        checkpoint = torch.load(checkpoint_path.as_posix())
        self.model.load_state_dict(checkpoint['model_state_dict'])

        test_metrics = self.test(test_loader)

        return test_metrics

def train_path_prediction(processed_plays: List[Dict]):
    config = {
        'max_defenders': 4,
        'embedding_dim': 64,
        'safety_encoder_dims': [128, 64],
        'receiver_encoder_dims': [128, 64],
        'ball_encoder_dims': [128, 64],
        'defender_encoder_dims': [64],
        'social_pooling_dim': 32,
        'lstm_hidden_dim': 128,
        'lstm_layers': 2,
        'lstm_dropout': 0.2,
        'decoder_dims': [64, 32],
        'dropout_rate': 0.3,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
        'batch_size': 32,
        'max_epochs': 100,
        'patience': 10
    }
    
    # Create dataset with padding
    dataset = PathPredictionDataset(processed_plays)
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize and train model
    trainer = PathPredictionTrainer(config)
    test_metrics = trainer.train(
        train_loader, val_loader, test_loader, 
        epochs=config['max_epochs'], 
        patience=config['patience']
    )
    
    # Save training curves and results
    plt.figure(figsize=(10, 6))
    plt.plot(trainer.train_losses, label='Train Loss')
    plt.plot(trainer.val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid()
    plt.savefig(MODELS_DIR / 'training_curve.png')
    plt.close()
    
    # Convert all NumPy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return obj
    
    results = {
        'config': config,
        'test_metrics': convert_to_serializable(test_metrics),
        'train_losses': convert_to_serializable(trainer.train_losses),
        'val_losses': convert_to_serializable(trainer.val_losses)
    }

    LOG.info(f"Training completed with {len(trainer.train_losses)} epochs")
    json_test_metrics = convert_to_serializable(test_metrics)
    LOG.info(f"Final Test Metrics: \n{json.dumps(json_test_metrics, indent=2)}")
    
    with open(MODELS_DIR / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    LOG.info("Training completed!")

def predict_path(processed_plays: List[Dict]) -> pd.DataFrame:
    LOG.info("Predicting paths using the trained Social LSTM model")
    # Load best model
    checkpoint_path = MODELS_DIR / 'best_model.pth'
    checkpoint = torch.load(checkpoint_path.as_posix())
    config = checkpoint['config']
    
    dataset = PathPredictionDataset(processed_plays)
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    
    model = SocialLSTMPathPrediction(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    all_predictions = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            predictions, _ = model(batch)
            predictions = predictions.cpu().numpy()
            sequence_mask = batch['sequence_mask'].cpu().numpy()
            frames = batch['frames'].cpu().numpy()
            
            for i in range(len(predictions)):
                valid_indices = sequence_mask[i] == 1
                for j in range(len(predictions[i])):
                    if valid_indices[j]:
                        all_predictions.append({
                            'gpid': processed_plays[i]['gpid'],
                            'frame_id': frames[i][j],
                            'nfl_id': processed_plays[i]['safety_nfl_id'],
                            'pred_x': predictions[i][j][0],
                            'pred_y': predictions[i][j][1]
                        })
    
    return pd.DataFrame(all_predictions)