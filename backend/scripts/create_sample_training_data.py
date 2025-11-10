"""
Create a sample training data file to fit the processor
This ensures the processor knows the expected feature structure
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import CLVDataProcessor

def create_sample_training_data(output_path='data/clv_features_sample.csv', n_samples=100):
    """Create a sample training data file with the correct structure."""
    
    processor = CLVDataProcessor()
    
    # Generate sample data using the processor's method
    df = processor.download_sample_data(n_samples=n_samples, random_state=42)
    
    # Save to file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Created sample training data: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Features (excluding target): {len(df.columns) - 1}")
    print(f"Columns: {list(df.columns)}")
    
    return df

if __name__ == '__main__':
    create_sample_training_data()

