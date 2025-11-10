"""
Customer Lifetime Value (CLV) Prediction Model
Deep Learning Architecture using PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLVPredictor(nn.Module):
    """
    Deep Neural Network for Customer Lifetime Value Prediction.
    
    Architecture:
    - Input Layer: Accepts customer features
    - Multiple Fully Connected Layers with Batch Normalization
    - Dropout for regularization
    - Output Layer: Single value prediction (CLV)
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 256, 128, 64], dropout_rate=0.3):
        """
        Initialize the CLV Predictor model.
        
        Args:
            input_dim (int): Number of input features
            hidden_dims (list): List of hidden layer dimensions
            dropout_rate (float): Dropout probability for regularization
        """
        super(CLVPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer (single value for CLV)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input features of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Predicted CLV values of shape (batch_size, 1)
        """
        return self.network(x)


class CLVLSTMPredictor(nn.Module):
    """
    LSTM-based model for sequential customer behavior prediction.
    Useful when customer data has temporal sequences.
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout_rate=0.3):
        """
        Initialize the LSTM-based CLV Predictor.
        
        Args:
            input_dim (int): Number of input features per time step
            hidden_dim (int): LSTM hidden dimension
            num_layers (int): Number of LSTM layers
            dropout_rate (float): Dropout probability
        """
        super(CLVLSTMPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Fully connected layers for final prediction
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        """
        Forward pass through the LSTM network.
        
        Args:
            x (torch.Tensor): Input sequences of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Predicted CLV values of shape (batch_size, 1)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last output from LSTM
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = self.fc1(last_output)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def get_model(model_type='feedforward', **kwargs):
    """
    Factory function to get a model instance.
    
    Args:
        model_type (str): Type of model ('feedforward' or 'lstm')
        **kwargs: Model-specific arguments
        
    Returns:
        nn.Module: Model instance
    """
    if model_type == 'feedforward':
        return CLVPredictor(**kwargs)
    elif model_type == 'lstm':
        return CLVLSTMPredictor(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

