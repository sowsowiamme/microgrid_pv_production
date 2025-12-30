import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class FeaturePreprocessor:
    """
    特征预处理类，专门针对光伏和负载功率预测项目
    处理特征工程、数据清洗、特征缩放等任务
    """

    def __init__(self,
                 numeric_features: List[str] = None,
                 categorical_features: List[str] = None,
                 target_columns: List[str] = None,
                 scaling_method: str = 'standard'):
        """
        初始化特征预处理器

        Args:
            numeric_features: 数值特征列名列表
            categorical_features: 分类特征列名列表
            target_columns: 目标变量列名列表
            scaling_method: 缩放方法 ('standard', 'minmax', 'none')
        """
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.target_columns = target_columns or []
        self.scaling_method = scaling_method

        # 初始化转换器
        self.scalers = {}
        self.imputers = {}

        self.logger = logging.getLogger(__name__)
        self.logger.info("FeaturePreprocessor 初始化完成")


    def create_lag_features(self,
                            data: pd.DataFrame,
                            columns:list, lag_number=24) -> pd.DataFrame:
        """
        创建滞后特征（对于时间序列预测非常重要）

        Args:
            data: 输入数据
            columns: 需要创建滞后特征的列
            lags: 滞后步长列表（例如：[1, 2, 3, 24] 表示1小时前、2小时前等）

        Returns:
            包含滞后特征的DataFrame
        """
        df = data.copy()

        for col in columns:
            for lag in range(1, 1+lag_number):
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

        self.logger.info(f"为 {len(columns)} 个列创建了 {len(columns) * lag_number} 个滞后特征")
        return df

    def handle_missing_values(self, data: pd.DataFrame, strategy: str = 'forward_fill') -> pd.DataFrame:
        """
        处理缺失值

        Args:
            data: 输入数据
            strategy: 处理策略 ('forward_fill', 'interpolate', 'drop')
            forward_fill 向前填充， drop丢弃缺失值，interpolate填充缺失值

        Returns:
            处理缺失值后的DataFrame
        """
        df = data.copy()

        initial_missing = df.isnull().sum().sum()

        if strategy == 'forward_fill':
            df = df.ffill().bfill()  # 先向前填充，再向后填充处理开头缺失
        elif strategy == 'interpolate':
            df = df.interpolate(method='linear')
        elif strategy == 'drop':
            df = df.dropna()

        final_missing = df.isnull().sum().sum()
        self.logger.info(f"缺失值处理: {initial_missing} -> {final_missing} (策略: {strategy})")

        return df

    def detect_and_handle_outliers(self,
                                   data: pd.DataFrame,
                                   method: str = 'iqr',
                                   columns: List[str] = None) -> pd.DataFrame:
        """
        检测和处理异常值

        Args:
            data: 输入数据
            method: 异常值检测方法 ('iqr', 'zscore')
            columns: 需要检查的列，如果为None则检查所有数值列

        Returns:
            处理异常值后的DataFrame
        """
        df = data.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        outliers_count = 0

        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # 将异常值替换为边界值
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                df.loc[outliers, col] = np.where(
                    df.loc[outliers, col] < lower_bound, lower_bound, upper_bound
                )
                outliers_count += outliers.sum()

            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > 3

                # 将异常值替换为均值
                df.loc[outliers, col] = df[col].mean()
                outliers_count += outliers.sum()

        self.logger.info(f"检测并处理了 {outliers_count} 个异常值 (方法: {method})")
        return df

    def fit_scalers(self, data: pd.DataFrame) -> None:
        """
        拟合特征缩放器（仅在训练数据上调用）

        Args:
            data: 训练数据
        """
        numeric_cols = self._get_numeric_columns(data)

        for col in numeric_cols:
            if self.scaling_method == 'standard':
                self.scalers[col] = StandardScaler()
            elif self.scaling_method == 'minmax':
                self.scalers[col] = MinMaxScaler()

            # 只在非空值上拟合
            non_null_data = data[col].dropna().values.reshape(-1, 1)
            if len(non_null_data) > 0:
                self.scalers[col].fit(non_null_data)

    def transform_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        应用特征转换（缩放、填充等）

        Args:
            data: 输入数据

        Returns:
            转换后的DataFrame
        """
        df = data.copy()
        numeric_cols = self._get_numeric_columns(df)

        # 应用缩放
        for col in numeric_cols:
            if col in self.scalers:
                non_null_mask = df[col].notna()
                if non_null_mask.any():
                    scaled_values = self.scalers[col].transform(
                        df.loc[non_null_mask, col].values.reshape(-1, 1)
                    )
                    df.loc[non_null_mask, col] = scaled_values.flatten()

        return df

    def _get_numeric_columns(self, data: pd.DataFrame) -> List[str]:
        """获取数值列（排除目标列和时间特征）"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        # 排除目标列
        numeric_cols = [col for col in numeric_cols if col not in self.target_columns]

        # 排除基础时间特征（保留周期性编码的特征）
        time_base_features = ['hour', 'day_of_week', 'day_of_month', 'month', 'quarter', 'year', 'week_of_year']
        numeric_cols = [col for col in numeric_cols if col not in time_base_features]

        return numeric_cols

    def create_radiation_features(self, data: pd.DataFrame,
                                  radiation_col: str = 'global_rad:W',
                                  pv_production_col: str = 'pv_production',
                                  elevation_col: str = 'sun_elevation:d') -> pd.DataFrame:
        """
        创建辐射相关的物理特征

        Args:
            data: 输入数据
            radiation_col: 辐射强度列名
            pv_production_col: 光伏发电量列名
            elevation_col: 太阳高度角列名

        Returns:
            添加了辐射特征的DataFrame
        """
        df = data.copy()

        # 1. 辐射变化率特征（加速度特征）
        if radiation_col in df.columns:
            # 一阶差分 - 辐射趋势（单位：W/h）
            df['global_rad_trend_1h'] = df[radiation_col].diff(1)

            # 二阶差分 - 辐射加速度（单位：W/h²）
            df['global_rad_acceleration_1h'] = df['global_rad_trend_1h'].diff(1)


        # 2. 光伏发电变化特征
        if pv_production_col in df.columns:
            # 光伏发电量变化（绝对值变化 + 1 避免除零）， 注意变化一定不能用当前值减前一个小时的值，这样会造成数据泄漏
            df['pv_change_1h'] = np.abs(df[pv_production_col].shift(1) - df[pv_production_col].shift(2)) + 1

        # 3. 日照时段特征
        if elevation_col in df.columns:
            # 基础日照判断（太阳高度角 > 2度）
            df['is_daylight'] = (df[elevation_col] >= 2).astype(int)
        return df

    def get_feature_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取特征摘要

        Args:
            data: 输入数据

        Returns:
            特征摘要字典
        """
        summary = {
            'total_features': len(data.columns),
            'numeric_features': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(data.select_dtypes(include=['object']).columns),
            'missing_values': data.isnull().sum().sum(),
            'feature_names': data.columns.tolist(),
            'shape': data.shape
        }

        return summary

