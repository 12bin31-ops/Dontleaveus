"""Data loading and preprocessing module."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from typing import Tuple, Dict, Any, List
from pathlib import Path


class DataLoader:
    """Handle data loading and preprocessing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV."""
        print(f"📂 Loading data from {filepath}...")
        self.data = pd.read_csv(filepath)
        print(f"✅ Data loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        print(f"\n📊 Dataset Overview:")
        print(self.data.head())
        print(f"\n📈 Data Info:")
        print(self.data.info())
        return self.data
    
    def explore_data(self) -> None:
        """Exploratory data analysis."""
        if self.data is None:
            raise ValueError("Load data first!")
        
        print("\n" + "="*50)
        print("📊 EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Missing values
        print("\n🔍 Missing Values:")
        missing = self.data.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("✅ No missing values")
        
        # Target variable distribution
        target = self.config['data']['target_column']
        print(f"\n🎯 Target Variable ({target}) Distribution:")
        print(self.data[target].value_counts())
        print(f"Class Balance: {self.data[target].value_counts(normalize=True)}")
        
        # Numerical features
        print("\n📊 Numerical Features Stats:")
        print(self.data.describe())
        
        # Categorical features
        categorical = self.data.select_dtypes(include=['object']).columns.tolist()
        if categorical:
            print(f"\n🏷️ Categorical Features ({len(categorical)}):")
            for col in categorical:
                print(f"  {col}: {self.data[col].nunique()} unique values")
    
    def preprocess(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess data: handle missing values, encode, scale.
        
        Returns:
            X, y arrays
        """
        if self.data is None:
            raise ValueError("Load data first!")
        
        print("\n" + "="*50)
        print("🔧 PREPROCESSING DATA")
        print("="*50)
        
        df = self.data.copy()
        target = self.config['data']['target_column']
        
        # Separate target
        y = df[target].copy()
        X = df.drop(columns=[target])
        
        # Handle missing values
        if self.config['preprocessing']['handle_missing'] == 'drop':
            X = X.dropna()
            y = y[X.index]
        
        # Identify column types
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"\n✅ Numerical columns: {len(numerical_cols)}")
        print(f"✅ Categorical columns: {len(categorical_cols)}")
        
        # Encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Encode target variable
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        self.label_encoders[target] = le
        
        # Remove outliers (optional)
        if self.config['preprocessing']['remove_outliers']:
            threshold = self.config['preprocessing']['outlier_threshold']
            initial_size = len(X)
            
            # Z-score based outlier detection
            from scipy import stats
            z_scores = np.abs(stats.zscore(X[numerical_cols]))
            mask = (z_scores < threshold).all(axis=1)
            X = X[mask]
            y = y[mask]
            
            removed = initial_size - len(X)
            if removed > 0:
                print(f"✅ Removed {removed} outliers")
        
        # Scale numerical features
        if self.config['preprocessing']['scale_numerical']:
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        
        self.feature_names = X.columns.tolist()
        print(f"\n✅ Preprocessing completed. Final shape: {X.shape}")
        
        return X, y
    
    def train_test_split(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Split data into train/test sets."""
        test_size = self.config['data']['test_size']
        random_state = self.config['data']['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        # Handle imbalanced data with SMOTE
        if self.config['data']['use_smote']:
            print("\n✅ Applying SMOTE for imbalanced data...")
            smote = SMOTE(random_state=random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"   After SMOTE - Train shape: {X_train.shape}")
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"\n✅ Train/Test Split:")
        print(f"   Train: {X_train.shape[0]} samples")
        print(f"   Test: {X_test.shape[0]} samples")
        print(f"   Train class distribution:\n{pd.Series(y_train).value_counts()}")
        print(f"   Test class distribution:\n{pd.Series(y_test).value_counts()}")
    
    def get_train_test_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get train/test data."""
        if self.X_train is None:
            raise ValueError("Run train_test_split first!")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self.feature_names
    
    def save_processed_data(self, filepath: str) -> None:
        """Save processed data."""
        if self.data is None:
            raise ValueError("No processed data to save!")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(filepath, index=False)
        print(f"✅ Processed data saved to {filepath}")
