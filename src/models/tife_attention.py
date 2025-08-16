"""
TiFe attention layer

    T x N
    ↓   ↓
T x d   N x d
      ↓
 (N + T) x d
      ↓
    T x N


to evaluate:
    - different number of extraction layers
    - less compression

tests:
    - check output size at each step

"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TiFeAttention(nn.Module):
    """
    attention layer across both time and feature boundaries
    followed by

    things to test:
    - layer normalization in different places
    - different batch sizes
    """
    def __init__(
        self,
        window_size: int, #T
        num_features: int, #N
        hidden_dim: int, # d
        batch_size: int,
        dropout: float = 0.1
    ):
        super(TiFeAttention, self).__init__()

        self.num_features = num_features
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        
        # Time extractor components
        self.time_query = nn.Linear(num_features, hidden_dim)
        self.time_key = nn.Linear(num_features, hidden_dim)
        self.time_value = nn.Linear(num_features, hidden_dim)
        
        # Feature extractor components
        self.feature_query = nn.Linear(window_size, hidden_dim)
        self.feature_key = nn.Linear(window_size, hidden_dim)
        self.feature_value = nn.Linear(window_size, hidden_dim)

        # Window attention components for cross-window interactions
        self.window_query = nn.Linear((window_size + num_features) * hidden_dim, hidden_dim)
        self.window_key = nn.Linear((window_size + num_features) * hidden_dim, hidden_dim)
        self.window_value = nn.Linear((window_size + num_features) * hidden_dim, hidden_dim)
        
        # Output projection for window attention
        self.window_output = nn.Linear(hidden_dim, (window_size + num_features) * hidden_dim)

        # ANN extractor for (T+N) x d -> T x N transformation
        self.time_feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(num_features)
        
        # Attention map storage for analysis
        self.time_attention_maps = []
        self.feature_attention_maps = []
        self.window_attention_maps = []

    def _feature_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feature attention: attend across feature dimension
        
        Args:
            x: Input tensor (batch_size, window_size, num_features)
            
        Returns:
            Feature-attended tensor of dimension (batch_size, num_features, d)
        """
        batch_size, seq_len, num_features = x.shape
        
        # Transpose to (b, N, T) for feature attention
        x_transposed = x.transpose(1, 2)
        
        # Compute queries, keys, values for feature attention
        Q_feature : torch.Tensor = self.feature_query(x_transposed)  # (B, F, H)
        K_feature : torch.Tensor = self.feature_key(x_transposed)    # (B, F, H)
        V_feature : torch.Tensor = self.feature_value(x_transposed)  # (B, F, H)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q_feature, K_feature.transpose(-2, -1))
        attention_scores = attention_scores / (self.hidden_dim ** 0.5)
        
        # Apply softmax to get attention weights
        feature_attention_weights = F.softmax(attention_scores, dim=-1)
        self.feature_attention_maps.append(feature_attention_weights.detach())
        
        # Apply dropout
        feature_attention_weights = self.dropout(feature_attention_weights)
        
        # Apply attention to values
        feature_attended = torch.matmul(feature_attention_weights, V_feature)
        
        # transpose to (batch_size, num_features, d)
        feature_attended = feature_attended.transpose(1, 2)
        
        return feature_attended
    
    def _time_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feature attention: attend across time dimension
        
        Args:
            x: Input tensor (batch_size, window_size, num_features)
            
        Returns:
            time-attended tensor of dimension (batch_size, window_size, d)
        """

        # compute Q, K, V
        Q_time : torch.Tensor = self.time_query(x)
        K_time : torch.Tensor = self.time_key(x)
        V_time : torch.Tensor = self.time_value(x)

        # compute attention scores
        attention_scores = torch.matmul(Q_time, K_time.transpose(-2,-1))
        attention_scores = attention_scores / (self.hidden_dim ** 0.5)

        # apply softmax and append to attention maps
        time_attention_weights = F.softmax(attention_scores, dim=-1)
        self.time_attention_maps.append(time_attention_weights.detach())

        # dropout
        time_attention_weights = self.dropout(time_attention_weights)

        # apply attention
        time_attended = torch.matmul(time_attention_weights, V_time)

        return time_attended
    
    def _time_feature_extraction(self, x: torch.Tensor) -> torch.Tensor:
        """
        ANN extraction of time and feature attended matrix

        Args:
            x: Input tensor of shape (batch_size, (window_size + num_features), hidden_dim)

        Returns:
            ANN extracted outputs of shape (batch_size, window_size, num_features)
        """
        
        # ANN extraction: (T+N) x d -> T x N
        ann_output : torch.Tensor = self.time_feature_extractor(x)  # (batch_size, T+N, 1)
        ann_output = ann_output.squeeze(-1)  # (batch_size, T+N)
        
        # Split back into time and feature parts
        time_part = ann_output[:, :self.window_size]  # (batch_size, T)
        feature_part = ann_output[:, self.window_size:]  # (batch_size, N)
        
        # Create T x N output by outer product
        output = torch.einsum('bt,bn->btn', time_part, feature_part)  # (batch_size, T, N)

        return output
    
    def _window_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Window attention: attend across batch dimension (different windows)
        
        Args:
            x: Input tensor (batch_size, T+N, d)
            
        Returns:
            Window-attended tensor of shape (batch_size, T+N, d)
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Flatten spatial and feature dimensions for window attention
        x_flat = x.view(batch_size, -1)  # (batch_size, (T+N)*d)
        
        # Compute Q, K, V for window attention across batch dimension
        Q_window = self.window_query(x_flat)  # (batch_size, hidden_dim)
        K_window = self.window_key(x_flat)    # (batch_size, hidden_dim)
        V_window = self.window_value(x_flat)  # (batch_size, hidden_dim)
        
        # Compute attention scores across windows (batch dimension)
        attention_scores = torch.matmul(Q_window, K_window.transpose(0, 1))  # (batch_size, batch_size)
        attention_scores = attention_scores / (self.hidden_dim ** 0.5)
        
        # Apply softmax to get attention weights
        window_attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, batch_size)
        self.window_attention_maps.append(window_attention_weights.detach())
        
        # Apply dropout
        window_attention_weights = self.dropout(window_attention_weights)
        
        # Apply attention to values
        window_attended : torch.Tensor = torch.matmul(window_attention_weights, V_window)  # (batch_size, hidden_dim)
        
        # Project back to original dimensions
        window_output : torch.Tensor = self.window_output(window_attended)  # (batch_size, (T+N)*d)
        
        # Reshape back to (batch_size, T+N, d)
        output = window_output.view(batch_size, seq_len, hidden_dim)
        
        return output

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through TiFE attention
        
        Args:
            x: Input tensor of shape (batch_size, window_size, num_features)
            
        Returns:
            Tuple of (output_tensor, attention_maps_dict)
        """

        batch_size, window_size, num_features = x.shape

        assert batch_size == self.batch_size; f"batch size {batch_size} != self.batch_size {self.batch_size}"
        assert window_size == self.window_size; f"batch size {window_size} != self.window_size {self.window_size}"
        assert num_features == self.num_features; f"batch size {num_features} != self.num_features {self.num_features}"

        # attention pass
        feature_attended = self._feature_attention(x) # (batch_size, N, d)
        time_attended = self._time_attention(x) # (batch_size, T, d)

        # concatenate time and feature representations
        concatenated = torch.cat([time_attended, feature_attended], dim=1)  # (batch_size, T+N, d)
        
        # apply window attention before time-feature extraction
        window_attended = self._window_attention(concatenated)  # (batch_size, T+N, d)
        
        # extract across time and feature dimensions
        time_feature_extracted = self._time_feature_extraction(window_attended)  # (batch_size, T, N)
        
        # return attention maps
        attention_maps = {
            'time_attention': self.time_attention_maps[-1],
            'feature_attention': self.feature_attention_maps[-1]
        }
        
        return time_feature_extracted, attention_maps