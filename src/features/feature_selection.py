import pandas as pd
import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import logging


class FeatureSelector:
    """
    特征筛选类，专门用于光伏预测项目的特征选择
    """

    def __init__(self, target_column: str = 'pv_production',
                 correlation_threshold: float = 0.5,
                 top_k_features: int = 30,
                 must_include: Optional[Sequence[str]] = None,
                 exclude_columns: Optional[Sequence[str]] = None,
                 exclude_patterns: Optional[Sequence[str]] = None):
        """
        初始化特征选择器

        Args:
            target_column: 目标变量列名
            correlation_threshold: 相关性阈值，低于此值的特征将被剔除
            top_k_features: 选择的前K个最重要的特征数量
            must_include: 即使相关性低也保留的业务关键特征
            exclude_columns: 明确排除的列，例如不可上线获得的字段
            exclude_patterns: 按子字符串排除的列名模式
        """
        self.target_column = target_column
        self.correlation_threshold = correlation_threshold
        self.top_k_features = top_k_features
        self.must_include = list(must_include or [])
        self.exclude_columns = set(exclude_columns or [])
        self.exclude_patterns = list(exclude_patterns or [])

        self.selected_features = []
        self.feature_importance: Dict[str, float] = {}
        self.feature_report_: pd.DataFrame = pd.DataFrame()
        self.correlation_matrix_: Optional[pd.DataFrame] = None
        self.target_correlation_: pd.Series = pd.Series(dtype=float)

        self.logger = logging.getLogger(__name__)
        self.logger.info("FeatureSelector 初始化完成")

    def fit(
        self,
        data: pd.DataFrame,
        plot: bool = False,
        save_report_path: Optional[str] = None,
    ) -> "FeatureSelector":
        """
        只在训练集上拟合特征选择规则。

        这样可以把原来手动看 correlation matrix 的过程固定下来，并避免把
        test data 的信息泄漏进特征选择。
        """
        target_correlation = self.correlation_analysis(data, plot=plot)

        candidate_features = [
            feature
            for feature, corr in target_correlation.items()
            if abs(corr) >= self.correlation_threshold
        ]
        candidate_features = candidate_features[: self.top_k_features]

        for feature in self.must_include:
            if (
                feature in data.columns
                and feature != self.target_column
                and feature not in candidate_features
                and not self._is_excluded(feature)
            ):
                candidate_features.append(feature)

        self.selected_features = candidate_features
        self.feature_importance = {
            feature: float(abs(self.target_correlation_.loc[feature]))
            for feature in self.selected_features
            if feature in self.target_correlation_.index
        }
        self._build_feature_report(data)

        if save_report_path:
            self.save_selected_features(save_report_path)

        self.logger.info(f"自动筛选出 {len(self.selected_features)} 个特征")
        return self

    def transform(self, data: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
        """
        使用 fit 阶段确定的特征列转换任意数据集。

        训练集、验证集、测试集都应该调用同一份 selected_features，保证列顺序一致。
        """
        if not self.selected_features:
            raise ValueError("请先调用 fit() 或 select_features_correlation() 完成特征选择")

        missing_columns = [col for col in self.selected_features if col not in data.columns]
        if missing_columns:
            raise ValueError(f"以下已选择特征在数据中不存在: {missing_columns}")

        columns_to_keep = list(self.selected_features)
        if include_target and self.target_column in data.columns:
            columns_to_keep.append(self.target_column)

        transformed_data = data[columns_to_keep].copy()
        self.logger.info(f"特征转换完成，最终数据形状: {transformed_data.shape}")
        return transformed_data

    def fit_transform(
        self,
        data: pd.DataFrame,
        plot: bool = False,
        include_target: bool = True,
        save_report_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """在训练集上拟合并返回筛选后的数据。"""
        self.fit(data, plot=plot, save_report_path=save_report_path)
        return self.transform(data, include_target=include_target)

    def correlation_analysis(
        self,
        data: pd.DataFrame,
        plot: bool = True,
        figsize: tuple = (15, 12),
        save_path: str = 'reports/figures/correlation_heatmap.png',
    ) -> pd.Series:
        """
        执行相关性分析，筛选与目标变量最相关的特征

        Args:
            data: 输入数据（应该已经完成特征工程）
            plot: 是否绘制相关性热力图
            figsize: 图形大小

        Returns:
            与目标变量相关性超过阈值的特征 Series，保留相关性正负号
        """
        # 确保目标列存在
        if self.target_column not in data.columns:
            raise ValueError(f"目标列 '{self.target_column}' 不存在于数据中")

        # 选择数值列
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column not in numeric_columns:
            raise ValueError(f"目标列 '{self.target_column}' 必须是数值列")

        # copy a new dataframe with the numeric_columns
        numeric_data = data[numeric_columns].copy()

        # 计算相关性矩阵
        correlation_matrix = numeric_data.corr()
        self.correlation_matrix_ = correlation_matrix

        # 获取与目标变量的相关性
        target_correlation = correlation_matrix[self.target_column].drop(labels=[self.target_column])
        target_correlation = target_correlation.dropna()
        target_correlation = target_correlation[
            [feature for feature in target_correlation.index if not self._is_excluded(feature)]
        ]
        target_correlation = target_correlation.reindex(
            target_correlation.abs().sort_values(ascending=False).index
        )
        self.target_correlation_ = target_correlation

        # 筛选相关性较高的特征
        high_corr_features = target_correlation[
            abs(target_correlation) >= self.correlation_threshold
        ]

        self.logger.info(f"找到 {len(high_corr_features)} 个与 '{self.target_column}' 相关性 >= {self.correlation_threshold} 的特征")

        # 绘制相关性热力图
        if plot:
            self._plot_correlation_heatmap(correlation_matrix, high_corr_features, figsize, save_path)

        return high_corr_features

    def _plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame,
                                  high_corr_features: pd.Series,
                                  figsize: tuple,
                                  save_path: str):
        """绘制相关性热力图"""
        # 选择要显示的特征（目标变量 + 高相关性特征）
        features_to_plot = [self.target_column] + high_corr_features.index.tolist()[: self.top_k_features]
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

        plt.title(f'Correlation Heatmap with "{self.target_column}"', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        # 保存图片
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"相关性热力图已保存到 '{output_path}'")

    def select_features_correlation(self, data: pd.DataFrame) -> List[str]:
        """
        基于相关性分析选择特征

        Args:
            data: 输入数据

        Returns:
            选择的特征列表
        """
        self.fit(data, plot=True)
        return self.selected_features

    def save_selected_features(self, path: str) -> None:
        """保存特征筛选结果，方便训练流程复用和审计。"""
        if not self.selected_features:
            raise ValueError("没有可保存的特征，请先调用 fit()")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'target_column': self.target_column,
            'correlation_threshold': self.correlation_threshold,
            'top_k_features': self.top_k_features,
            'must_include': self.must_include,
            'exclude_columns': sorted(self.exclude_columns),
            'exclude_patterns': self.exclude_patterns,
            'selected_features': self.selected_features,
            'feature_report': self.feature_report_.to_dict('records'),
        }

        with output_path.open('w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        self.logger.info(f"特征筛选结果已保存到: {output_path}")

    def load_selected_features(self, path: str) -> List[str]:
        """加载之前保存的特征列表。"""
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        self.selected_features = list(payload['selected_features'])
        self.target_column = payload.get('target_column', self.target_column)
        self.correlation_threshold = payload.get('correlation_threshold', self.correlation_threshold)
        self.top_k_features = payload.get('top_k_features', self.top_k_features)
        self.must_include = payload.get('must_include', self.must_include)
        self.exclude_columns = set(payload.get('exclude_columns', self.exclude_columns))
        self.exclude_patterns = payload.get('exclude_patterns', self.exclude_patterns)
        self.feature_report_ = pd.DataFrame(payload.get('feature_report', []))

        self.logger.info(f"已加载 {len(self.selected_features)} 个特征: {path}")
        return self.selected_features

    def _build_feature_report(self, data: pd.DataFrame) -> None:
        """生成特征筛选报告，帮助解释为什么某个特征被选中。"""
        records = []
        for feature, corr in self.target_correlation_.items():
            records.append({
                'feature': feature,
                'correlation': float(corr),
                'abs_correlation': float(abs(corr)),
                'missing_rate': float(data[feature].isna().mean()) if feature in data.columns else np.nan,
                'selected': feature in self.selected_features,
                'must_include': feature in self.must_include,
            })

        self.feature_report_ = pd.DataFrame(records)

    def _is_excluded(self, feature: str) -> bool:
        """判断特征是否被显式排除。"""
        if feature == self.target_column:
            return True
        if feature in self.exclude_columns:
            return True
        return any(pattern in feature for pattern in self.exclude_patterns)