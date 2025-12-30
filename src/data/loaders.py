"""
Data loading and preprocessing utilities for time series PV data.
Handles proper time-based train/validation/test splits.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging

class DataLoader:
    """Handles data loading and preprocessing for PV prediction with enhanced flexibility."""

    def __init__(self, daily_points: int = 24, time_column: str = 'time',
                 target_columns: Optional[list] = None):
        """
        Initialize DataLoader with configurable parameters.

        Args:
            daily_points: Expected number of data points per day (default 24 for hourly data)
            time_column: Name of the time column in the dataset
            target_columns: List of target column names for prediction
        """
        self.daily_points = daily_points # here, we have 24 points per day for hourly data, in other dataset, we could also have 1 point per 15 minutes
        self.time_column = time_column
        self.target_columns = target_columns or []
        self.logger = logging.getLogger(__name__)

    def load_data(self, file_path: str,
                  sep: str = ',',
                  encoding: str = 'utf-8',
                  filter_complete_days: bool = True) -> pd.DataFrame:
        """
        Load and preprocess data from CSV file.

        Args:
            file_path: Path to the CSV file
            sep: Column separator
            encoding: File encoding
            filter_complete_days: Whether to filter only complete days

        Returns:
            Preprocessed DataFrame with datetime index
        """
        try:
            # Load data
            dataset = pd.read_csv(file_path, sep=sep, encoding=encoding)
            self.logger.info(f"Successfully loaded data from {file_path}, shape: {dataset.shape}")

            # Validate required columns
            self._validate_columns(dataset.columns)

            # Preprocess data
            dataset = self._preprocess_data(dataset, filter_complete_days)

            return dataset

        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def _validate_columns(self, columns: pd.Index) -> None:
        """Validate that required columns exist in the dataset."""
        if self.time_column not in columns:
            raise ValueError(f"Time column '{self.time_column}' not found in dataset")

        if self.target_columns:
            missing_targets = [col for col in self.target_columns if col not in columns]
            if missing_targets:
                self.logger.warning(f"Missing target columns: {missing_targets}")

    def _preprocess_data(self, dataset: pd.DataFrame, filter_complete_days: bool) -> pd.DataFrame:
        """Main preprocessing pipeline."""
        # Create a copy to avoid modifying original data
        dataset_processed = dataset.copy()

        # Parse datetime
        dataset_processed[self.time_column] = pd.to_datetime(dataset_processed[self.time_column])

        # Add date column for grouping
        dataset_processed['date'] = dataset_processed[self.time_column].dt.date

        # Filter complete days if requested
        if filter_complete_days:
            dataset_processed = self._filter_complete_days(dataset_processed)

        # Sort and set index
        dataset_processed.sort_values(self.time_column, inplace=True)
        dataset_processed.set_index(self.time_column, inplace=True)

        # Basic data quality check
        self._data_quality_report(dataset_processed)

        return dataset_processed

    def _filter_complete_days(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Filter dataset to include only complete days."""
        daily_counts = dataset.groupby('date')[self.time_column].count()
        complete_days = daily_counts[daily_counts == self.daily_points].index

        self.logger.info(f"Found {len(complete_days)} complete days out of {len(daily_counts)} total days")

        if len(complete_days) == 0:
            self.logger.warning("No complete days found! Returning original dataset")
            return dataset

        return dataset[dataset['date'].isin(complete_days)]

    def _data_quality_report(self, dataset: pd.DataFrame) -> None:
        """Generate basic data quality report."""
        report = {
            "total_records": len(dataset),
            "date_range": f"{dataset.index.min()} to {dataset.index.max()}",
            "missing_values": dataset.isnull().sum().sum(),
            "numeric_columns": dataset.select_dtypes(include=[np.number]).columns.tolist()
        }

        self.logger.info("Data Quality Report:")
        for key, value in report.items():
            self.logger.info(f"  {key}: {value}")

    def get_feature_target_split(self, dataset: pd.DataFrame,
                                 feature_columns: list,
                                 target_columns: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into features and targets.

        Args:
            dataset: Input DataFrame
            feature_columns: List of feature column names
            target_columns: List of target column names

        Returns:
            Tuple of (features, targets) DataFrames
        """
        features = dataset[feature_columns].copy()
        targets = dataset[target_columns].copy()

        self.logger.info(f"Feature shape: {features.shape}, Target shape: {targets.shape}")
        return features, targets

    def create_time_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from datetime index.

        Args:
            dataset: Input DataFrame with datetime index

        Returns:
            DataFrame with added time features
        """
        dataset_with_features = dataset.copy()

        # Extract time components
        dataset_with_features['hour'] = dataset.index.hour
        dataset_with_features['day_of_week'] = dataset.index.dayofweek
        dataset_with_features['month'] = dataset.index.month
        dataset_with_features['day_of_year'] = dataset.index.dayofyear


        self.logger.info("Added time-based features")
        return dataset_with_features