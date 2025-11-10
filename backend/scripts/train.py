"""
Training script for CLV prediction model
"""

import sys
import os
import argparse
import torch

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.clv_model import get_model
from data.data_loader import CLVDataProcessor
from utils.trainer import CLVTrainer


def main():
    parser = argparse.ArgumentParser(description='Train CLV Prediction Model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to dataset CSV file (optional, will generate sample data if not provided)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Proportion of test set')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Proportion of validation set')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='feedforward',
                       choices=['feedforward', 'lstm'],
                       help='Type of model architecture')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 256, 128, 64],
                       help='Hidden layer dimensions for feedforward model')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for regularization')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Save arguments
    parser.add_argument('--save_path', type=str, default='models/saved_model.pth',
                       help='Path to save trained model')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_state)
        torch.cuda.manual_seed_all(args.random_state)
    
    print("="*60)
    print("Customer Lifetime Value (CLV) Prediction Model Training")
    print("="*60)
    
    # Load and preprocess data
    print("\n[1/4] Loading and preprocessing data...")
    processor = CLVDataProcessor(data_path=args.data_path)
    train_loader, val_loader, test_loader, input_dim, feature_names = processor.process_pipeline(
        file_path=args.data_path,
        test_size=args.test_size,
        val_size=args.val_size,
        batch_size=args.batch_size,
        random_state=args.random_state
    )
    
    print(f"Input dimension: {input_dim}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print("\n[2/4] Creating model...")
    if args.model_type == 'feedforward':
        model = get_model(
            model_type='feedforward',
            input_dim=input_dim,
            hidden_dims=args.hidden_dims,
            dropout_rate=args.dropout_rate
        )
    else:
        model = get_model(
            model_type='lstm',
            input_dim=input_dim,
            hidden_dim=args.hidden_dims[0] if args.hidden_dims else 128,
            num_layers=2,
            dropout_rate=args.dropout_rate
        )
    
    print(f"Model type: {args.model_type}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n[3/4] Training model...")
    trainer = CLVTrainer(model, device='cuda' if torch.cuda.is_available() else 'cpu')
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_path=args.save_path
    )
    
    # Evaluate on test set
    print("\n[4/4] Evaluating on test set...")
    from utils.evaluator import CLVEvaluator
    evaluator = CLVEvaluator(model)
    test_metrics = evaluator.evaluate(test_loader)
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print(f"Model saved to: {args.save_path}")
    print("="*60)


if __name__ == '__main__':
    main()

