#!/usr/bin/env python3
"""
Download sample dataset for testing.

This script downloads the Kaggle Telco Customer Churn dataset.

Note: You'll need kaggle API credentials:
1. Visit https://www.kaggle.com/settings/account
2. Download your kaggle.json file
3. Place it in ~/.kaggle/
"""

import os
import sys
from pathlib import Path

def download_kaggle_dataset():
    """Download from Kaggle using kaggle CLI."""
    try:
        import kaggle
    except ImportError:
        print("❌ kaggle-api not installed. Install with: pip install kaggle")
        return False
    
    dataset = "blastchar/telco-customer-churn"
    output_dir = "data/raw"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"📥 Downloading {dataset}...")
        kaggle.api.dataset_download_files(dataset, path=output_dir, unzip=True)
        print("✅ Dataset downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
        return False


def download_from_url():
    """Download from public URL as fallback."""
    import urllib.request
    import zipfile
    
    url = "https://github.com/IBM/telco-customer-churn-on-icp4d/raw/master/data/Telco-Customer-Churn.csv"
    output_file = "data/raw/churn_data.csv"
    
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"📥 Downloading from {url}...")
        urllib.request.urlretrieve(url, output_file)
        print(f"✅ Dataset saved to {output_file}")
        return True
    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
        return False


def create_dummy_data():
    """Create dummy data for testing."""
    import pandas as pd
    import numpy as np
    
    print("📊 Generating dummy customer data...")
    
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'customerID': [f'CUST_{i:05d}' for i in range(n_samples)],
        'tenure': np.random.randint(0, 72, n_samples),
        'MonthlyCharges': np.random.uniform(20, 118, n_samples),
        'TotalCharges': np.random.uniform(100, 8000, n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    }
    
    # Generate churn based on features (realistic pattern)
    churn_prob = (
        0.2 * (data['tenure'] < 12) +
        0.15 * (np.array(data['MonthlyCharges']) > 100) +
        0.1 * (np.array(data['Contract'] == 'Month-to-month']) > 0)
    )
    
    data['Churn'] = (np.random.random(n_samples) < churn_prob).astype(int)
    
    df = pd.DataFrame(data)
    
    output_file = "data/raw/churn_data.csv"
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"✅ Dummy data created: {output_file}")
    print(f"   Shape: {df.shape}")
    print(f"   Churn rate: {df['Churn'].mean():.2%}")
    
    return True


def main():
    """Main function."""
    print("\n" + "="*60)
    print("📥 DATASET SETUP")
    print("="*60)
    
    # Try methods in order
    methods = [
        ("Kaggle API", download_kaggle_dataset),
        ("Public URL", download_from_url),
        ("Generate Dummy Data", create_dummy_data)
    ]
    
    for method_name, method_func in methods:
        print(f"\n🔄 Trying {method_name}...")
        if method_func():
            print(f"\n✅ Setup complete using {method_name}")
            return 0
    
    print("\n❌ All download methods failed")
    return 1


if __name__ == '__main__':
    sys.exit(main())
