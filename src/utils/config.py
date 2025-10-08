"""
Configuration management for the photovoltaic prediction system.
Centralized configuration using YAML files.
"""

import yaml
from typing import Dict, Any
import os


class Config:
    """Central configuration management for PV prediction system."""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Config file {self.config_path} not found. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration for quick start."""
        return {
            'models': {
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'random_state': 42
                },
                'lightgbm': {
                    'n_estimators': 1000,
                    'learning_rate': 0.05,
                    'max_depth': 8,
                    'num_leaves': 31,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'verbose': -1
                },
                'xgboost': {
                    'n_estimators': 1000,
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                }
            },
            'data': {
                'train_start': '2020-01-02',
                'train_end': '2020-12-31',
                'val_start': '2021-01-01', 
                'val_end': '2021-01-31',
                'test_start': '2021-02-01',
                'test_end': '2021-03-07'
            },
            'features': {
                'pv_features': [
                    'hour', 'day_of_year', 'month',
                    'sun_elevation:d', 'global_rad:W', 
                    'clear_sky_rad:W', 'clear_rad_ratio',
                    'temp', 'total_cloud_cover:p',
                    'pv_lag_24h'
                ]
            }
        }
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for specific model."""
        return self.config['models'].get(model_name, {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data splitting configuration."""
        return self.config['data']
    
    def get_feature_config(self) -> list:
        """Get feature list for PV prediction."""
        return self.config['features'].get('pv_features', [])
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        self.config.update(updates)
    
    def save_config(self, path: str = None):
        """Save current configuration to file."""
        save_path = path or self.config_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)


# Global configuration instance
config = Config()
