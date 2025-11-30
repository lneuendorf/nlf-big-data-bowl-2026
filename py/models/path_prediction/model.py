from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOG = logging.getLogger(__name__)

MODELS_DIR = Path("/Users/lukeneuendorf/projects/nfl-big-data-bowl-2026/data/models/path_prediction/")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class PathPredictionDataset(Dataset):
    """Dataset class for path prediction"""
    def __init__(self, processed_plays: List[Dict]):
        self.processed_plays = processed_plays
        
    def __len__(self):
        return len(self.processed_plays)
    
    def __getitem__(self, idx):
        play = self.processed_plays[idx]
        
        return {
            'safety': torch.FloatTensor(play['safety']),
            'receiver': torch.FloatTensor(play['receiver']),
            'ball': torch.FloatTensor(play['ball']),
            'defenders': torch.FloatTensor(play['defenders']),
            'defender_mask': torch.FloatTensor(play['mask']),
            'globals': torch.FloatTensor(play['globals']),
            'target': torch.FloatTensor(play['target']),
            'frames': torch.LongTensor(play['frames']),
            'sequence_mask': torch.ones(len(play['frames'])),  # All frames are valid
        }

class SocialLSTMPathPrediction(nn.Module):
    """
    Social LSTM for safety path prediction with dynamic sequence and defender masking
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
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
        
        # Regularization
        self.dropout = nn.Dropout(config.get('dropout_rate', 0.2))
        
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
        global_encoded = self.global_encoder(globals_feat.unsqueeze(-1))
        global_encoded = global_encoded.unsqueeze(1).expand(-1, seq_len, -1)
        
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
    """Training and evaluation class"""
    
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
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions, _ = self.model(batch)
            
            # Compute masked loss
            target = batch['target']
            sequence_mask = batch['sequence_mask']
            
            # Calculate loss only on valid sequence positions
            loss_per_element = self.criterion(predictions, target)
            masked_loss = (loss_per_element * sequence_mask.unsqueeze(-1)).sum()
            valid_positions = sequence_mask.sum() * 2  # *2 for (x,y) coordinates
            
            if valid_positions > 0:
                loss = masked_loss / valid_positions
            else:
                loss = masked_loss
                
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
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                predictions, _ = self.model(batch)
                
                target = batch['target']
                sequence_mask = batch['sequence_mask']
                
                loss_per_element = self.criterion(predictions, target)
                masked_loss = (loss_per_element * sequence_mask.unsqueeze(-1)).sum()
                valid_positions = sequence_mask.sum() * 2
                
                if valid_positions > 0:
                    loss = masked_loss / valid_positions
                else:
                    loss = masked_loss
                    
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches if num_batches > 0 else 0
    
    def test(self, test_loader: DataLoader) -> Dict:
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                predictions, _ = self.model(batch)
                target = batch['target']
                sequence_mask = batch['sequence_mask']
                
                # Calculate loss
                loss_per_element = self.criterion(predictions, target)
                masked_loss = (loss_per_element * sequence_mask.unsqueeze(-1)).sum()
                valid_positions = sequence_mask.sum() * 2
                
                if valid_positions > 0:
                    loss = masked_loss / valid_positions
                else:
                    loss = masked_loss
                    
                total_loss += loss.item()
                num_batches += 1
                
                # Store predictions and targets for additional metrics
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        # Calculate additional metrics
        predictions_array = np.concatenate(all_predictions)
        targets_array = np.concatenate(all_targets)
        
        # Calculate RMSE
        mse_per_coord = np.mean((predictions_array - targets_array) ** 2, axis=0)
        rmse_per_coord = np.sqrt(mse_per_coord)
        overall_rmse = np.sqrt(np.mean((predictions_array - targets_array) ** 2))
        
        # Calculate MAE
        mae_per_coord = np.mean(np.abs(predictions_array - targets_array), axis=0)
        overall_mae = np.mean(np.abs(predictions_array - targets_array))
        
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
            LOG.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            LOG.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
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
        LOG.info(f"Test Metrics: {test_metrics}")
        
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
    
    # Create dataset and split
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
    
    results = {
        'config': config,
        'test_metrics': test_metrics,
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses
    }

    LOG.info(f"Final Test Metrics: {test_metrics}")
    LOG.info(f"Training Losses: {trainer.train_losses}")
    LOG.info(f"Validation Losses: {trainer.val_losses}")
    
    with open(MODELS_DIR / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    LOG.info("Training completed!")