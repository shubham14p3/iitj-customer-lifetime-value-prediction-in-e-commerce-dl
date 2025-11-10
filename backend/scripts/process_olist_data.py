"""
Script to process Olist e-commerce dataset and create CLV features
This script processes the Olist Brazilian e-commerce dataset to extract
features for Customer Lifetime Value prediction.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_olist_dataset_info():
    """
    Print information about downloading the Olist dataset.
    The Olist dataset can be downloaded from Kaggle.
    """
    print("="*70)
    print("OLIST DATASET DOWNLOAD INSTRUCTIONS")
    print("="*70)
    print("\nThe Olist dataset is available on Kaggle:")
    print("URL: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce")
    print("\nRequired files for CLV prediction:")
    print("  1. olist_customers_dataset.csv (you already have this)")
    print("  2. olist_orders_dataset.csv")
    print("  3. olist_order_items_dataset.csv")
    print("  4. olist_order_payments_dataset.csv")
    print("\nAfter downloading:")
    print("  1. Extract all CSV files to a 'data/olist/' directory")
    print("  2. Run this script: python scripts/process_olist_data.py")
    print("="*70)


def calculate_clv_features(customers_df, orders_df, order_items_df, payments_df):
    """
    Calculate CLV-related features from Olist dataset.
    
    Args:
        customers_df: DataFrame with customer information
        orders_df: DataFrame with order information
        order_items_df: DataFrame with order items and prices
        payments_df: DataFrame with payment information
        
    Returns:
        DataFrame with CLV features and target CLV value
    """
    print("Calculating CLV features from Olist dataset...")
    
    # Convert date columns
    orders_df['order_purchase_timestamp'] = pd.to_datetime(orders_df['order_purchase_timestamp'])
    orders_df['order_delivered_customer_date'] = pd.to_datetime(orders_df['order_delivered_customer_date'])
    
    # Calculate order values
    order_values = order_items_df.groupby('order_id').agg({
        'price': 'sum',
        'freight_value': 'sum'
    }).reset_index()
    order_values['order_total'] = order_values['price'] + order_values['freight_value']
    
    # Merge orders with values
    orders_with_values = orders_df.merge(order_values, on='order_id', how='left')
    orders_with_values['order_total'] = orders_with_values['order_total'].fillna(0)
    
    # Filter to delivered orders only
    delivered_orders = orders_with_values[
        orders_with_values['order_status'] == 'delivered'
    ].copy()
    
    # Calculate customer-level features
    customer_features = []
    
    for customer_id in customers_df['customer_unique_id'].unique():
        customer_orders = delivered_orders[
            delivered_orders['customer_id'].isin(
                customers_df[customers_df['customer_unique_id'] == customer_id]['customer_id'].values
            )
        ]
        
        if len(customer_orders) == 0:
            continue
        
        # Basic purchase metrics
        total_orders = len(customer_orders)
        total_revenue = customer_orders['order_total'].sum()
        avg_order_value = customer_orders['order_total'].mean()
        
        # Time-based features
        first_order_date = customer_orders['order_purchase_timestamp'].min()
        last_order_date = customer_orders['order_purchase_timestamp'].max()
        days_as_customer = (last_order_date - first_order_date).days
        days_since_first_purchase = (datetime.now() - first_order_date).days if pd.notna(first_order_date) else 0
        days_since_last_purchase = (datetime.now() - last_order_date).days if pd.notna(last_order_date) else 0
        
        # Frequency metrics
        if days_as_customer > 0:
            purchase_frequency = total_orders / (days_as_customer / 30)  # orders per month
        else:
            purchase_frequency = total_orders
        
        # Payment method diversity
        customer_payments = payments_df[
            payments_df['order_id'].isin(customer_orders['order_id'])
        ]
        payment_methods = customer_payments['payment_type'].nunique()
        
        # Order status distribution
        order_statuses = customer_orders['order_status'].value_counts()
        
        # Geographic features
        customer_info = customers_df[customers_df['customer_unique_id'] == customer_id].iloc[0]
        
        # Calculate CLV (simplified: total revenue + predicted future value)
        # Using a simple formula: historical revenue + (avg_order_value * expected_future_orders)
        # Expected future orders based on frequency and recency
        if days_since_last_purchase < 90:  # Active customer
            expected_future_months = 12
        elif days_since_last_purchase < 180:  # At-risk customer
            expected_future_months = 6
        else:  # Inactive customer
            expected_future_months = 2
        
        expected_future_orders = purchase_frequency * expected_future_months
        predicted_future_value = avg_order_value * expected_future_orders * 0.7  # Discount factor
        clv = total_revenue + predicted_future_value
        
        # Create feature row
        features = {
            'customer_unique_id': customer_id,
            # Purchase behavior
            'total_orders': total_orders,
            'total_revenue': total_revenue,
            'avg_order_value': avg_order_value,
            'days_as_customer': days_as_customer,
            'days_since_first_purchase': days_since_first_purchase,
            'days_since_last_purchase': days_since_last_purchase,
            'purchase_frequency': purchase_frequency,
            # Payment features
            'payment_methods_count': payment_methods,
            # Geographic features (encoded)
            'customer_state': customer_info.get('customer_state', 'UNKNOWN'),
            'customer_city': customer_info.get('customer_city', 'UNKNOWN'),
            # Target variable
            'clv': clv
        }
        
        customer_features.append(features)
    
    features_df = pd.DataFrame(customer_features)
    
    # Add additional derived features
    features_df['revenue_per_day'] = features_df['total_revenue'] / (features_df['days_as_customer'] + 1)
    features_df['order_value_std'] = customer_orders.groupby('customer_id')['order_total'].std().values if len(customer_orders) > 0 else 0
    
    print(f"Calculated features for {len(features_df)} customers")
    return features_df


def process_olist_dataset(data_dir='data/olist', output_path='data/clv_features.csv'):
    """
    Main function to process Olist dataset.
    
    Args:
        data_dir: Directory containing Olist CSV files
        output_path: Path to save processed features
    """
    print("="*70)
    print("Processing Olist Dataset for CLV Prediction")
    print("="*70)
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"\nError: Data directory '{data_dir}' not found!")
        print("\nPlease download the Olist dataset and extract to 'data/olist/'")
        download_olist_dataset_info()
        return None
    
    # Required files
    required_files = {
        'customers': 'olist_customers_dataset.csv',
        'orders': 'olist_orders_dataset.csv',
        'order_items': 'olist_order_items_dataset.csv',
        'payments': 'olist_order_payments_dataset.csv'
    }
    
    # Check if files exist
    missing_files = []
    file_paths = {}
    for key, filename in required_files.items():
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            file_paths[key] = file_path
            print(f"✓ Found {filename}")
        else:
            missing_files.append(filename)
            print(f"✗ Missing {filename}")
    
    if missing_files:
        print(f"\nError: Missing required files: {', '.join(missing_files)}")
        download_olist_dataset_info()
        return None
    
    # Load datasets
    print("\nLoading datasets...")
    try:
        customers_df = pd.read_csv(file_paths['customers'])
        orders_df = pd.read_csv(file_paths['orders'])
        order_items_df = pd.read_csv(file_paths['order_items'])
        payments_df = pd.read_csv(file_paths['payments'])
        print("✓ All datasets loaded successfully")
    except Exception as e:
        print(f"✗ Error loading datasets: {e}")
        return None
    
    # Calculate features
    print("\nProcessing data...")
    features_df = calculate_clv_features(customers_df, orders_df, order_items_df, payments_df)
    
    # Save processed data
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    features_df.to_csv(output_path, index=False)
    print(f"\n✓ Processed features saved to: {output_path}")
    print(f"  Total customers: {len(features_df)}")
    print(f"  Features: {len(features_df.columns) - 1}")  # Excluding target
    print(f"  Target: CLV")
    
    return features_df


def create_sample_from_customers(customers_csv='olist_customers_dataset.csv', 
                                  output_path='data/clv_features_sample.csv'):
    """
    Create a sample dataset with synthetic CLV features based on customer data.
    This is used when only the customers CSV is available.
    
    Args:
        customers_csv: Path to customers CSV file
        output_path: Path to save sample features
    """
    print("="*70)
    print("Creating Sample CLV Dataset from Customer Data")
    print("="*70)
    print("\nNote: This creates synthetic CLV features based on customer data.")
    print("For real CLV prediction, please download the full Olist dataset.")
    
    if not os.path.exists(customers_csv):
        print(f"\nError: {customers_csv} not found!")
        return None
    
    print(f"\nLoading {customers_csv}...")
    customers_df = pd.read_csv(customers_csv)
    print(f"✓ Loaded {len(customers_df)} customers")
    
    # Generate synthetic features based on customer data
    np.random.seed(42)
    
    # Use customer state and city as features
    state_counts = customers_df['customer_state'].value_counts()
    city_counts = customers_df['customer_city'].value_counts()
    
    features_list = []
    
    for idx, row in customers_df.iterrows():
        # Generate features based on customer location (as proxy for behavior)
        state = row['customer_state']
        city = row['customer_city']
        
        # Use state/city as seed for consistent synthetic data
        state_rank = state_counts.get(state, 0)
        city_rank = city_counts.get(city, 0)
        
        # Generate synthetic purchase behavior
        np.random.seed(hash(f"{state}_{city}") % 2**32)
        
        total_orders = np.random.poisson(3 + state_rank % 5)
        avg_order_value = np.random.lognormal(3.5 + (state_rank % 3) * 0.2, 0.8)
        days_as_customer = np.random.randint(30, 365 * 2)
        days_since_first_purchase = days_as_customer + np.random.randint(0, 180)
        days_since_last_purchase = np.random.randint(0, min(180, days_as_customer))
        purchase_frequency = total_orders / (days_as_customer / 30) if days_as_customer > 0 else 0
        payment_methods_count = np.random.randint(1, 4)
        
        # Calculate synthetic CLV
        total_revenue = total_orders * avg_order_value
        if days_since_last_purchase < 90:
            expected_future_months = 12
        elif days_since_last_purchase < 180:
            expected_future_months = 6
        else:
            expected_future_months = 2
        
        expected_future_orders = purchase_frequency * expected_future_months
        predicted_future_value = avg_order_value * expected_future_orders * 0.7
        clv = total_revenue + predicted_future_value
        
        features = {
            'customer_unique_id': row['customer_unique_id'],
            'total_orders': total_orders,
            'total_revenue': total_revenue,
            'avg_order_value': avg_order_value,
            'days_as_customer': days_as_customer,
            'days_since_first_purchase': days_since_first_purchase,
            'days_since_last_purchase': days_since_last_purchase,
            'purchase_frequency': purchase_frequency,
            'payment_methods_count': payment_methods_count,
            'customer_state': state,
            'customer_city': city,
            'clv': clv
        }
        
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    
    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    features_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Sample features saved to: {output_path}")
    print(f"  Total customers: {len(features_df)}")
    print(f"  Features: {len(features_df.columns) - 1}")
    print(f"\nNote: This is synthetic data. For real predictions, use the full Olist dataset.")
    
    return features_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Olist dataset for CLV prediction')
    parser.add_argument('--data_dir', type=str, default='data/olist',
                       help='Directory containing Olist CSV files')
    parser.add_argument('--output', type=str, default='data/clv_features.csv',
                       help='Output path for processed features')
    parser.add_argument('--customers_only', action='store_true',
                       help='Create sample dataset from customers CSV only')
    parser.add_argument('--customers_csv', type=str, default='olist_customers_dataset.csv',
                       help='Path to customers CSV (for --customers_only mode)')
    
    args = parser.parse_args()
    
    if args.customers_only:
        create_sample_from_customers(args.customers_csv, args.output)
    else:
        process_olist_dataset(args.data_dir, args.output)

