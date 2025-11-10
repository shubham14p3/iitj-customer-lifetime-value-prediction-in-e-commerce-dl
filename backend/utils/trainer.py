"""
Training utilities and functions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import os


class CLVTrainer:
    """
    Trainer class for CLV prediction model.
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize trainer.
        
        Args:
            model (nn.Module): PyTorch model
            device (str): Device to train on
        """
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader, criterion, optimizer):
        """
        Train for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
            criterion: Loss function
            optimizer: Optimizer
            
        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for features, targets in tqdm(train_loader, desc="Training"):
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = self.model(features)
            loss = criterion(predictions, targets.unsqueeze(1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, val_loader, criterion):
        """
        Validate model.
        
        Args:
            val_loader (DataLoader): Validation data loader
            criterion: Loss function
            
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for features, targets in tqdm(val_loader, desc="Validating"):
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(features)
                loss = criterion(predictions, targets.unsqueeze(1))
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_loader, val_loader, num_epochs=50, learning_rate=0.001,
              weight_decay=1e-5, patience=10, save_path='models/saved_model.pth'):
        """
        Complete training loop.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            num_epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay for regularization
            patience (int): Early stopping patience
            save_path (str): Path to save best model
            
        Returns:
            dict: Training history
        """
        # Setup loss, optimizer, and scheduler
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Create directory for saving model
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader, criterion)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping and model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save model configuration for easier loading
                model_config = {
                    'model_type': 'feedforward' if hasattr(self.model, 'input_dim') else 'lstm',
                    'input_dim': self.model.input_dim if hasattr(self.model, 'input_dim') else None,
                    'hidden_dims': self.model.hidden_dims if hasattr(self.model, 'hidden_dims') else None,
                    'dropout_rate': self.model.dropout_rate if hasattr(self.model, 'dropout_rate') else None,
                }
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'model_config': model_config,
                }, save_path)
                print(f"Model saved with validation loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Load best model
        checkpoint = torch.load(save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nTraining completed. Best validation loss: {best_val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }

