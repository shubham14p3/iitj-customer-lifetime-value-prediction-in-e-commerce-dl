"""
Evaluation utilities
"""

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


class CLVEvaluator:
    """
    Evaluator class for CLV prediction model.
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize evaluator.
        
        Args:
            model (nn.Module): Trained PyTorch model
            device (str): Device to evaluate on
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def predict(self, data_loader):
        """
        Make predictions on a dataset.
        
        Args:
            data_loader (DataLoader): Data loader
            
        Returns:
            tuple: (predictions, targets) as numpy arrays
        """
        predictions = []
        targets = []
        
        with torch.no_grad():
            for features, target in data_loader:
                features = features.to(self.device)
                pred = self.model(features)
                
                predictions.extend(pred.cpu().numpy())
                targets.extend(target.numpy())
        
        return np.array(predictions).flatten(), np.array(targets)
    
    def evaluate(self, data_loader, verbose=True):
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader (DataLoader): Data loader
            verbose (bool): Print metrics
            
        Returns:
            dict: Evaluation metrics
        """
        predictions, targets = self.predict(data_loader)
        
        # Calculate metrics
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
        if verbose:
            print("\n" + "="*50)
            print("Evaluation Metrics:")
            print("="*50)
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")
            print(f"R² Score: {r2:.4f}")
            print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
            print("="*50)
        
        return metrics
    
    def plot_predictions(self, data_loader, save_path=None, max_samples=1000):
        """
        Plot predictions vs actual values.
        
        Args:
            data_loader (DataLoader): Data loader
            save_path (str): Path to save plot (optional)
            max_samples (int): Maximum number of samples to plot
        """
        predictions, targets = self.predict(data_loader)
        
        # Limit samples for visualization
        if len(predictions) > max_samples:
            indices = np.random.choice(len(predictions), max_samples, replace=False)
            predictions = predictions[indices]
            targets = targets[indices]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(targets, predictions, alpha=0.5)
        plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
        plt.xlabel('Actual CLV')
        plt.ylabel('Predicted CLV')
        plt.title('Predicted vs Actual CLV')
        plt.grid(True, alpha=0.3)
        
        # Add R² to plot
        r2 = r2_score(targets, predictions)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_residuals(self, data_loader, save_path=None, max_samples=1000):
        """
        Plot residual distribution.
        
        Args:
            data_loader (DataLoader): Data loader
            save_path (str): Path to save plot (optional)
            max_samples (int): Maximum number of samples to plot
        """
        predictions, targets = self.predict(data_loader)
        residuals = targets - predictions
        
        # Limit samples for visualization
        if len(residuals) > max_samples:
            indices = np.random.choice(len(residuals), max_samples, replace=False)
            residuals = residuals[indices]
        
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Residuals (Actual - Predicted)')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')
        plt.grid(True, alpha=0.3)
        plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()

