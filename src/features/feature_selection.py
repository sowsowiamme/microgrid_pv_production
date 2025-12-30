import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import logging


class FeatureSelector:
    """
    特征筛选类，专门用于光伏预测项目的特征选择
    """

    def __init__(self, target_column: str = 'pv_production',
                 correlation_threshold: float = 0.1,
                 top_k_features: int = 30):
        """
        初始化特征选择器

        Args:
            target_column: 目标变量列名
            correlation_threshold: 相关性阈值，低于此值的特征将被剔除
            top_k_features: 选择的前K个最重要的特征数量
        """
        self.target_column = target_column
        self.correlation_threshold = correlation_threshold
        self.top_k_features = top_k_features
        self.selected_features = []
        self.feature_importance = {}

        self.logger = logging.getLogger(__name__)
        self.logger.info("FeatureSelector 初始化完成")

    def correlation_analysis(self, data: pd.DataFrame,
                             plot: bool = True,
                             figsize: tuple = (15, 12)) -> pd.DataFrame:
        """
        执行相关性分析，筛选与目标变量最相关的特征

        Args:
            data: 输入数据（应该已经完成特征工程）
            plot: 是否绘制相关性热力图
            figsize: 图形大小

        Returns:
            相关性矩阵DataFrame
        """
        # 确保目标列存在
        if self.target_column not in data.columns:
            raise ValueError(f"目标列 '{self.target_column}' 不存在于数据中")

        # 选择数值列
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        if self.target_column not in numeric_columns:
            numeric_columns.append(self.target_column)

        numeric_data = data[numeric_columns].copy()

        # 计算相关性矩阵
        correlation_matrix = numeric_data.corr()

        # 获取与目标变量的相关性
        target_correlation = correlation_matrix[self.target_column].sort_values(ascending=False)

        # 筛选相关性较高的特征
        high_corr_features = target_correlation[
            (abs(target_correlation) >= self.correlation_threshold) &
            (target_correlation.index != self.target_column)
            ]

        self.logger.info(f"找到 {len(high_corr_features)} 个与 '{self.target_column}' 相关性 >= {self.correlation_threshold} 的特征")

        # 绘制相关性热力图
        if plot:
            self._plot_correlation_heatmap(correlation_matrix, high_corr_features, figsize)

        return high_corr_features

    def _plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame,
                                  high_corr_features: pd.Series, figsize: tuple):
        """绘制相关性热力图"""
        # 选择要显示的特征（目标变量 + 高相关性特征）
        features_to_plot = [self.target_column] + high_corr_features.index.tolist()
        plot_corr_matrix = correlation_matrix.loc[features_to_plot, features_to_plot]

        plt.figure(figsize=figsize)

        # 创建热力图
        mask = np.triu(np.ones_like(plot_corr_matrix, dtype=bool))  # 创建上三角掩码
        sns.heatmap(plot_corr_matrix,
                    mask=mask,
                    annot=True,
                    cmap='RdBu_r',
                    center=0,
                    fmt='.2f',
                    cbar_kws={'shrink': 0.8})

        plt.title(f'特征与 "{self.target_column}" 的相关性热力图', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        # 保存图片
        os.makedirs('reports/figures', exist_ok=True)
        plt.savefig('reports/figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

        self.logger.info("相关性热力图已保存到 'reports/figures/correlation_heatmap.png'")

    def select_features_correlation(self, data: pd.DataFrame) -> List[str]:
        """
        基于相关性分析选择特征

        Args:
            data: 输入数据

        Returns:
            选择的特征列表
        """
        high_corr_features = self.correlation_analysis(data, plot=True)
        self.selected_features = high_corr_features.index.tolist()

        # 从selected_features中移除目标变量（如果存在）
        if self.target_column in self.selected_features:
            self.selected_features.remove(self.target_column)

        self.logger.info(f"基于相关性选择了 {len(self.selected_features)} 个特征")
        return self.selected_features

    # def transform(self, data: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     应用特征选择，返回只包含选择特征的数据
    #
    #     Args:
    #         data: 输入数据
    #
    #     Returns:
    #         只包含选择特征和目标变量的DataFrame
    #     """
    #     if not self.selected_features:
    #         raise ValueError("请先运行特征选择方法")
    #
    #     # 确保目标列存在
    #     columns_to_keep = self.selected_features.copy()
    #     if self.target_column in data.columns:
    #         columns_to_keep.append(self.target_column)
    #
    #     # 检查所有选择的列是否都存在
    #     missing_columns = [col for col in columns_to_keep if col not in data.columns]
    #     if missing_columns:
    #         self.logger.warning(f"以下选择的特征在数据中不存在: {missing_columns}")
    #         columns_to_keep = [col for col in columns_to_keep if col in data.columns]
    #
    #     transformed_data = data[columns_to_keep].copy()
    #
    #     self.logger.info(f"特征选择完成，最终数据形状: {transformed_data.shape}")
    #     return transformed_data