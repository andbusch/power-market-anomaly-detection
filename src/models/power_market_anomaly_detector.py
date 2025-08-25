"""
Power market anomaly detector

1. TiFe attention layer 
2. encoder
3. decoder
4. TiFe attention layer

"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple
from tife_attention import TiFeAttention

class AnomalyDetector(nn.Module):
    def __init__(
        self,
        window_size: int, #T
        num_features: int, #N
        hidden_dim: int, # d
        batch_size: int,
        dropout: float = 0.1
    ):
        super(AnomalyDetector, self).__init__()

        # model params
        self.num_features = num_features
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size


        # model attributes
        self.initial_attention = TiFeAttention(window_size, num_features, hidden_dim, batch_size, dropout)
        
        # Encoder-decoder architecture for anomaly detection
        self.encoder_decoder = nn.Sequential(
            # Encoder: compress input to bottleneck representation
            nn.Linear(num_features, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Bottleneck layer (compressed representation)
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Decoder: reconstruct from bottleneck to original dimensions
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_features)
        )
        
        self.final_attention = TiFeAttention(window_size, num_features, hidden_dim, batch_size, dropout)
        
        # Anomaly scoring layers
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass through the anomaly detector
        
        Args:
            x: Input tensor of shape (batch_size, window_size, num_features)
            
        Returns:
            Tuple of (reconstructed_output, anomaly_scores, reconstruction_error, attention_maps)
        """
        # Initial attention processing
        initial_attention_out, attention_maps_1 = self.initial_attention(x)
        
        # Encoder-decoder reconstruction
        batch_size, window_size, num_features = initial_attention_out.shape
        
        # Apply encoder-decoder to each time step
        reconstructed = []
        for t in range(window_size):
            time_step = initial_attention_out[:, t, :]  # (batch_size, num_features)
            reconstructed_step = self.encoder_decoder(time_step)  # (batch_size, num_features)
            reconstructed.append(reconstructed_step.unsqueeze(1))
        
        reconstructed_output = torch.cat(reconstructed, dim=1)  # (batch_size, window_size, num_features)
        
        # Final attention processing
        final_attention_out, attention_maps_2 = self.final_attention(reconstructed_output)
        
        # Compute reconstruction error
        reconstruction_error = F.mse_loss(final_attention_out, x, reduction='none')  # (batch_size, window_size, num_features)
        reconstruction_error = reconstruction_error.mean(dim=-1)  # (batch_size, window_size)
        
        # Compute anomaly scores for each time step
        anomaly_scores = []
        for t in range(window_size):
            time_step_error = reconstruction_error[:, t].unsqueeze(-1)  # (batch_size, 1)
            # Expand to match feature dimension for anomaly scorer input
            time_step_features = final_attention_out[:, t, :]  # (batch_size, num_features)
            anomaly_score = self.anomaly_scorer(time_step_features)  # (batch_size, 1)
            anomaly_scores.append(anomaly_score)
        
        anomaly_scores = torch.cat(anomaly_scores, dim=1)  # (batch_size, window_size)
        
        # Combine attention maps from both attention layers
        combined_attention_maps = {
            'initial_attention': attention_maps_1,
            'final_attention': attention_maps_2
        }
        
        return final_attention_out, anomaly_scores, reconstruction_error, combined_attention_maps
    
    def detect_anomalies(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Detect anomalies in the input data
        
        Args:
            x: Input tensor of shape (batch_size, window_size, num_features)
            threshold: Anomaly threshold (default: 0.5)
            
        Returns:
            Binary anomaly predictions (batch_size, window_size)
        """
        with torch.no_grad():
            _, anomaly_scores, _, _ = self.forward(x)
            anomaly_predictions = (anomaly_scores > threshold).float()
            return anomaly_predictions
    
    def get_reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss for training
        
        Args:
            x: Input tensor of shape (batch_size, window_size, num_features)
            
        Returns:
            Scalar reconstruction loss
        """
        reconstructed_output, _, reconstruction_error, _ = self.forward(x)
        return reconstruction_error.mean()
