"""
Data loading and preprocessing utilities for time series PV data.
Handles proper time-based train/validation/test splits.
"""

import pandas as pd
from typing import Tuple
from src.utils.config import config


class DataLoader:
    """Handles data loading and time-based splitting for PV prediction."""
    
    def __init__(self):
        self.data_config = config.get_data_config()
    
    def load_csv(self, file_path: str, time_col: str = 'time') -> pd.DataFrame:
        """
        Load CSV file and parse datetime index.
        
        Args:
            file_path: Path to CSV file
            time_col: Name of timestamp column
            
        Returns:
            DataFrame with datetime index, sorted chronologically
        """
        df = pd.read_csv(file_path, parse_dates=[time_col])
        df.set_index(time_col, inplace=True)
        df.sort_index(inplace=True)
        return df
    
    def split_data(self, df: pd.DataFrame, target_col: str) -> Tuple:
        """
        Split data into train/validation/test sets based on configured time periods.
        
        Args:
            df: Input DataFrame with datetime index
            target_col: Name of target variable column (e.g., 'pv_production')
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("Splitting data based on time periods...")
        
        # Training set
        train_mask = (df.index >= self.data_config['train_start']) & (df.index <= self.data_config['train_end'])
        X_train = df[train_mask]
        y_train = X_train[target_col]
        X_train = X_train.drop(columns=[target_col])
        
        # Validation set  
        val_mask = (df.index >= self.data_config['val_start']) & (df.index <= self.data_config['val_end'])
        X_val = df[val_mask]
        y_val = X_val[target_col]
        X_val = X_val.drop(columns=[target_col])
        
        # Test set
        test_mask = (df.index >= self.data_config['test_start']) & (df.index <= self.data_config['test_end'])
        X_test = df[test_mask]
        y_test = X_test[target_col]
        X_test = X_test.drop(columns=[target_col])
        
        print(f"Data splits created:")
        print(f"  Training: {X_train.shape[0]} samples ({X_train.index.min()} to {X_train.index.max()})")
        print(f"  Validation: {X_val.shape[0]} samples ({X_val.index.min()} to {X_val.index.max()})")
        print(f"  Test: {X_test.shape[0]} samples ({X_test.index.min()} to {X_test.index.max()})")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_feature_subset(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """
        Extract specific features from DataFrame.
        
        Args:
            df: Input DataFrame
            features: List of feature names to extract
            
        Returns:
            DataFrame with only the specified features
        """
        available_features = [f for f in features if f in df.columns]
        missing_features = [f for f in features if f not in df.columns]
        
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
        
        return df[available_features]
