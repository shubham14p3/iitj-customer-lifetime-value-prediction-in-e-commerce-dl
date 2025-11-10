"""
Inference script for CLV prediction
"""

import sys
import os
import argparse
import torch
import pandas as pd
import numpy as np

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.clv_model import get_model
from data.data_loader import CLVDataProcessor
from utils.evaluator import CLVEvaluator


def load_model(model_path, model_type=None, input_dim=None, 
               hidden_dims=None, dropout_rate=None, device='cpu'):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path (str): Path to model checkpoint
        model_type (str): Type of model (optional, will use checkpoint if available)
        input_dim (int): Input dimension (optional, will use checkpoint if available)
        hidden_dims (list): Hidden layer dimensions (optional, will use checkpoint if available)
        dropout_rate (float): Dropout rate (optional, will use checkpoint if available)
        device (str): Device to load model on
        
    Returns:
        nn.Module: Loaded model
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Try to get config from checkpoint
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        model_type = model_type or config.get('model_type', 'feedforward')
        input_dim = input_dim or config.get('input_dim')
        hidden_dims = hidden_dims or config.get('hidden_dims', [128, 256, 128, 64])
        dropout_rate = dropout_rate or config.get('dropout_rate', 0.3)
    else:
        # Fallback to defaults if not in checkpoint
        model_type = model_type or 'feedforward'
        hidden_dims = hidden_dims or [128, 256, 128, 64]
        dropout_rate = dropout_rate or 0.3
    
    # Check if input_dim is available
    if input_dim is None:
        raise ValueError("input_dim must be provided either as argument or in checkpoint.")
    
    # Create model
    if model_type == 'feedforward':
        model = get_model(
            model_type='feedforward',
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        )
    else:
        model = get_model(
            model_type='lstm',
            input_dim=input_dim,
            hidden_dim=hidden_dims[0] if hidden_dims else 128,
            num_layers=2,
            dropout_rate=dropout_rate
        )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def predict_single_customer(model, processor, customer_data, device='cpu'):
    """
    Predict CLV for a single customer.
    
    Args:
        model: Trained model
        processor: Data processor with fitted scaler
        customer_data (dict or pd.Series): Customer features
        device: Device to run inference on
        
    Returns:
        float: Predicted CLV
    """
    # Convert to DataFrame if needed
    if isinstance(customer_data, dict):
        df = pd.DataFrame([customer_data])
    elif isinstance(customer_data, pd.Series):
        df = pd.DataFrame([customer_data])
    else:
        df = customer_data
    
    # Preprocess (assuming target column doesn't exist)
    if 'clv' in df.columns:
        df = df.drop(columns=['clv'])
    
    # Handle categorical variables
    for col in processor.label_encoders:
        if col in df.columns:
            # Handle unseen categories
            try:
                df[col] = processor.label_encoders[col].transform(df[col].astype(str))
            except ValueError:
                # If unseen category, use most common category
                df[col] = 0
    
    # Ensure all columns are present
    if processor.feature_names:
        for col in processor.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[processor.feature_names]
    
    # Convert to numpy and scale
    features = df.values.astype(float)
    features = processor.scaler.transform(features)
    
    # Predict
    features_tensor = torch.FloatTensor(features).to(device)
    with torch.no_grad():
        prediction = model(features_tensor)
    
    return prediction.cpu().item()


def main():
    parser = argparse.ArgumentParser(description='CLV Prediction Inference')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, default='feedforward',
                       choices=['feedforward', 'lstm'],
                       help='Type of model architecture')
    parser.add_argument('--input_dim', type=int, default=None,
                       help='Input dimension (required if not in checkpoint)')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 256, 128, 64],
                       help='Hidden layer dimensions')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to dataset for evaluation')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for batch inference')
    
    # Output arguments
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save predictions (CSV)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate prediction plots')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load processor (needed for preprocessing)
    processor = CLVDataProcessor()
    
    # If data_path provided, load it to get input_dim and fit processor
    if args.data_path:
        df = processor.load_data(args.data_path)
        features, targets, feature_names = processor.preprocess_data(df)
        input_dim = features.shape[1]
    elif args.input_dim:
        input_dim = args.input_dim
    else:
        raise ValueError("Either --data_path or --input_dim must be provided")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(
        args.model_path,
        model_type=args.model_type,
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout_rate,
        device=device
    )
    print("Model loaded successfully!")
    
    # If data_path provided, evaluate
    if args.data_path:
        print("\nEvaluating on dataset...")
        from torch.utils.data import DataLoader
        from data.data_loader import CLVDataset
        
        # Create test dataloader
        test_dataset = CLVDataset(features, targets)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Evaluate
        evaluator = CLVEvaluator(model, device=device)
        metrics = evaluator.evaluate(test_loader)
        
        # Make predictions
        predictions, actuals = evaluator.predict(test_loader)
        
        # Save predictions if output_path provided
        if args.output_path:
            results_df = pd.DataFrame({
                'actual_clv': actuals,
                'predicted_clv': predictions,
                'error': actuals - predictions,
                'abs_error': np.abs(actuals - predictions)
            })
            results_df.to_csv(args.output_path, index=False)
            print(f"\nPredictions saved to {args.output_path}")
        
        # Generate plots if requested
        if args.plot:
            print("\nGenerating plots...")
            evaluator.plot_predictions(test_loader, save_path='predictions_plot.png')
            evaluator.plot_residuals(test_loader, save_path='residuals_plot.png')
    
    else:
        # Example single prediction
        print("\nExample single customer prediction:")
        example_customer = {
            'age': 35,
            'gender': 'M',
            'total_orders': 10,
            'avg_order_value': 50.0,
            'days_since_first_purchase': 365,
            'days_since_last_purchase': 30,
            'total_page_views': 100,
            'avg_session_duration': 300.0,
            'bounce_rate': 0.3,
            'return_rate': 0.1,
            'category_diversity': 5,
            'premium_category_ratio': 0.4,
            'email_opens': 20,
            'email_clicks': 5,
            'promo_code_usage': 1,
            'customer_segment': 'B'
        }
        
        # Note: This requires a fitted processor
        # In practice, you'd save the processor along with the model
        print("Note: Single prediction requires a fitted processor.")
        print("For production use, save and load the processor along with the model.")


if __name__ == '__main__':
    main()

