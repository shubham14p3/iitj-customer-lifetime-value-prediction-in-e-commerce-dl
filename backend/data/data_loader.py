"""
Data Loading and Preprocessing Utilities
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')


class CLVDataset(Dataset):
    """
    PyTorch Dataset for CLV prediction.
    """
    
    def __init__(self, features, targets):
        """
        Initialize dataset.
        
        Args:
            features (np.ndarray): Feature matrix
            targets (np.ndarray): Target CLV values
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class CLVDataProcessor:
    """
    Data processor for CLV prediction dataset.
    Handles downloading, preprocessing, and splitting data.
    """
    
    def __init__(self, data_path=None):
        """
        Initialize data processor.
        
        Args:
            data_path (str): Path to dataset file (optional)
        """
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.feature_means = None  # Store mean values for missing feature imputation
        
    def download_sample_data(self, n_samples=10000, random_state=42):
        """
        Generate synthetic e-commerce customer data for demonstration.
        In production, replace this with actual data loading.
        
        Args:
            n_samples (int): Number of samples to generate
            random_state (int): Random seed for reproducibility
            
        Returns:
            pd.DataFrame: Generated dataset
        """
        np.random.seed(random_state)
        
        # Generate synthetic features
        data = {
            # Customer demographics
            'age': np.random.randint(18, 80, n_samples),
            'gender': np.random.choice(['M', 'F', 'Other'], n_samples),
            
            # Purchase behavior
            'total_orders': np.random.poisson(5, n_samples),
            'avg_order_value': np.random.lognormal(3.5, 0.8, n_samples),
            'days_since_first_purchase': np.random.randint(0, 365*3, n_samples),
            'days_since_last_purchase': np.random.randint(0, 180, n_samples),
            
            # Engagement metrics
            'total_page_views': np.random.poisson(50, n_samples),
            'avg_session_duration': np.random.lognormal(4.0, 0.7, n_samples),
            'bounce_rate': np.random.beta(2, 5, n_samples),
            'return_rate': np.random.beta(3, 7, n_samples),
            
            # Product categories
            'category_diversity': np.random.randint(1, 10, n_samples),
            'premium_category_ratio': np.random.beta(2, 3, n_samples),
            
            # Marketing
            'email_opens': np.random.poisson(10, n_samples),
            'email_clicks': np.random.poisson(2, n_samples),
            'promo_code_usage': np.random.binomial(1, 0.3, n_samples),
            
            # Customer segment
            'customer_segment': np.random.choice(['A', 'B', 'C', 'D'], n_samples, p=[0.1, 0.3, 0.4, 0.2]),
        }
        
        df = pd.DataFrame(data)
        
        # Calculate CLV as target (with some realistic relationship)
        df['clv'] = (
            df['total_orders'] * df['avg_order_value'] * 0.3 +
            df['days_since_first_purchase'] * 0.01 +
            df['total_page_views'] * 0.5 +
            df['avg_session_duration'] * 0.2 +
            df['category_diversity'] * 10 +
            np.random.normal(0, 50, n_samples)
        )
        df['clv'] = np.maximum(df['clv'], 0)  # Ensure non-negative
        
        return df
    
    def load_data(self, file_path=None):
        """
        Load data from file or generate sample data.
        
        Args:
            file_path (str): Path to CSV file (optional)
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        if file_path:
            if os.path.exists(file_path):
                return pd.read_csv(file_path)
            else:
                print(f"File not found: {file_path}. Generating sample data...")
                return self.download_sample_data()
        elif self.data_path and os.path.exists(self.data_path):
            return pd.read_csv(self.data_path)
        else:
            print("No data file provided. Generating sample data...")
            return self.download_sample_data()
    
    def preprocess_data(self, df, target_col='clv'):
        """
        Preprocess data for model training.
        
        Args:
            df (pd.DataFrame): Raw dataset
            target_col (str): Name of target column
            
        Returns:
            tuple: (features, targets, feature_names)
        """
        df = df.copy()
        
        # Separate features and target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        targets = df[target_col].values
        features_df = df.drop(columns=[target_col])
        
        # Exclude identifier columns (customer_id, customer_unique_id, etc.)
        id_columns = [col for col in features_df.columns 
                     if 'id' in col.lower() or col.lower() == 'customer_unique_id']
        if id_columns:
            print(f"Excluding identifier columns: {id_columns}")
            features_df = features_df.drop(columns=id_columns)
        
        # Handle categorical variables
        categorical_cols = features_df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            features_df[col] = self.label_encoders[col].fit_transform(features_df[col].astype(str))
        
        # Convert to numpy
        features = features_df.values.astype(float)
        self.feature_names = features_df.columns.tolist()
        
        # Store feature means for imputation of missing features in transform_data
        self.feature_means = features_df.mean().to_dict()
        
        # Scale features
        features = self.scaler.fit_transform(features)
        
        return features, targets, self.feature_names
    
    def transform_data(self, df, target_col='clv', require_target=False):
        """
        Transform data using already fitted processor.
        Ensures feature compatibility with training data.
        
        Args:
            df (pd.DataFrame): Raw dataset
            target_col (str): Name of target column
            require_target (bool): Whether target is required
            
        Returns:
            tuple: (features, targets, feature_names)
        """
        df = df.copy()
        
        # Check if target exists
        has_target = target_col in df.columns
        if require_target and not has_target:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        # Get targets if available
        targets = df[target_col].values if has_target else np.zeros(len(df))
        
        # Separate features
        features_df = df.drop(columns=[target_col]) if has_target else df.copy()
        
        # Exclude identifier columns
        id_columns = [col for col in features_df.columns 
                     if 'id' in col.lower() or col.lower() == 'customer_unique_id']
        if id_columns:
            features_df = features_df.drop(columns=id_columns)
        
        # Ensure all expected features are present
        if self.feature_names is None:
            raise ValueError("Processor not fitted. Please fit the processor first.")
        
        # Track missing features for warning
        missing_features = []
        present_features = []
        
        # Add missing columns with default values (use mean from training data if available)
        for col in self.feature_names:
            if col not in features_df.columns:
                missing_features.append(col)
                # Use mean from training data if available, otherwise 0
                default_value = self.feature_means.get(col, 0) if self.feature_means else 0
                features_df[col] = default_value
            else:
                present_features.append(col)
        
        # Warn if too many features are missing (indicates data structure mismatch)
        if len(missing_features) > len(present_features):
            import warnings
            warnings.warn(
                f"Warning: {len(missing_features)} out of {len(self.feature_names)} required features are missing. "
                f"Missing: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}. "
                f"Present: {present_features[:5]}{'...' if len(present_features) > 5 else ''}. "
                f"Predictions may be inaccurate. Please ensure your data has the expected features.",
                UserWarning
            )
        
        # Remove extra columns not in training data
        features_df = features_df[self.feature_names]
        
        # Handle categorical variables
        for col in self.feature_names:
            if col in self.label_encoders:
                # Transform categorical columns
                try:
                    features_df[col] = self.label_encoders[col].transform(features_df[col].astype(str))
                except ValueError:
                    # Handle unseen categories - use most common (0)
                    features_df[col] = 0
        
        # Convert to numpy
        features = features_df.values.astype(float)
        
        # Scale features (using fitted scaler)
        features = self.scaler.transform(features)
        
        return features, targets, self.feature_names
    
    def create_dataloaders(self, features, targets, test_size=0.2, val_size=0.1, 
                          batch_size=64, random_state=42):
        """
        Create train, validation, and test dataloaders.
        
        Args:
            features (np.ndarray): Feature matrix
            targets (np.ndarray): Target values
            test_size (float): Proportion of test set
            val_size (float): Proportion of validation set (from training set)
            batch_size (int): Batch size for dataloaders
            random_state (int): Random seed
            
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=test_size, random_state=random_state
        )
        
        # Split train into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=random_state
        )
        
        # Create datasets
        train_dataset = CLVDataset(X_train, y_train)
        val_dataset = CLVDataset(X_val, y_val)
        test_dataset = CLVDataset(X_test, y_test)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def process_pipeline(self, file_path=None, test_size=0.2, val_size=0.1, 
                        batch_size=64, random_state=42):
        """
        Complete data processing pipeline.
        
        Args:
            file_path (str): Path to data file (optional)
            test_size (float): Proportion of test set
            val_size (float): Proportion of validation set
            batch_size (int): Batch size
            random_state (int): Random seed
            
        Returns:
            tuple: (train_loader, val_loader, test_loader, input_dim, feature_names)
        """
        # Load data
        df = self.load_data(file_path)
        
        # Preprocess
        features, targets, feature_names = self.preprocess_data(df)
        input_dim = features.shape[1]
        
        # Create dataloaders
        train_loader, val_loader, test_loader = self.create_dataloaders(
            features, targets, test_size, val_size, batch_size, random_state
        )
        
        return train_loader, val_loader, test_loader, input_dim, feature_names

