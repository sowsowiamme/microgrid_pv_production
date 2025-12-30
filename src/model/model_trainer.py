import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import joblib
import json
import yaml
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 机器学习相关
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import (train_test_split, TimeSeriesSplit,
                                     cross_val_score, GridSearchCV, RandomizedSearchCV)
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, explained_variance_score)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

# 设置matplotlib后端
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logging.warning("XGBoost not available")

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    logging.warning("LightGBM not available")

class ModelTrainer:
    """
    灵活的模型训练器，支持多种模型和超参数调优
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化模型训练器

        Args:
            config_path: 模型配置文件的路径
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("ModelTrainer 初始化完成")
        # 加载配置
        if config_path is None:
            # 默认配置路径
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "model_config.yaml"

        self.config = self._load_config(config_path)
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()



    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载模型配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"配置已从 {config_path} 加载")
            return config
        except Exception as e:
            self.logger.error(f"加载配置失败: {str(e)}")
            # 返回默认配置
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'models': {
                'random_forest': {
                    'enabled': True,
                    'class': 'sklearn.ensemble.RandomForestRegressor',
                    'param_grid': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [10, 20, None]
                    }
                }
            },
            'training': {
                'test_size': 0.2,
                'random_state': 42,
                'cv_folds': 5
            }
        }

    def prepare_data(self, train_data: pd.DataFrame, test_data:pd.DataFrame,selected_columns:list, target_column: str) -> Tuple:
        """
        准备训练数据

        Args:
            data: 包含特征和目标变量的完整数据集
            target_column: 目标变量列名

        Returns:
            (X_train, X_test, y_train, y_test, feature_columns)
        """
        if target_column not in train_data.columns or target_column not in test_data.columns:
            raise ValueError(f"目标列 '{target_column}' 不存在于数据中")

        # 确定特征列
        # feature_columns = [col for col in data.columns if col != target_column]

        X_train = train_data[selected_columns].copy()
        y_train = train_data[target_column]
        X_test = test_data[selected_columns].copy()
        y_test = test_data[target_column]

        # # 分割数据
        # test_size = self.config['training'].get('test_size', 0.2)
        # random_state = self.config['training'].get('random_state', 42)
        #
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=test_size, random_state=random_state, shuffle=False
        # )

        self.logger.info(f"数据准备完成: 训练集={X_train.shape}, 测试集={X_test.shape}")
        return X_train, X_test, y_train, y_test, selected_columns

    def _get_cv_strategy(self, n_splits: int = 5):
        """获取交叉验证策略"""
        cv_strategy = self.config['training'].get('cv_strategy', 'timeseries')

        if cv_strategy == 'timeseries':
            return TimeSeriesSplit(n_splits=n_splits)
        else:
            return n_splits  # 使用默认的KFold

    def _instantiate_model(self, model_config: Dict[str, Any]) -> BaseEstimator:
        """实例化模型"""
        model_class_path = model_config['class']

        # 根据类路径实例化模型
        if model_class_path == "sklearn.ensemble.RandomForestRegressor":
            return RandomForestRegressor(random_state=42)
        elif model_class_path == "sklearn.ensemble.GradientBoostingRegressor":
            return GradientBoostingRegressor(random_state=42)
        elif model_class_path == "sklearn.linear_model.LinearRegression":
            return LinearRegression()
        elif model_class_path == "sklearn.linear_model.Ridge":
            return Ridge(random_state=42)
        elif model_class_path == "sklearn.linear_model.Lasso":
            return Lasso(random_state=42)
        elif model_class_path == "sklearn.svm.SVR":
            return SVR()
        elif model_class_path == "sklearn.neighbors.KNeighborsRegressor":
            return KNeighborsRegressor()
        elif model_class_path == "xgboost.XGBRegressor" and XGB_AVAILABLE:
            return xgb.XGBRegressor(random_state=42)
        elif model_class_path == "lightgbm.LGBMRegressor" and LGBM_AVAILABLE:
            return lgb.LGBMRegressor(random_state=42)
        else:
            raise ValueError(f"不支持的模型类: {model_class_path}")

    def train_models(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        训练所有启用的模型

        Args:
            data: 训练数据
            target_column: 目标变量列名

        Returns:
            训练结果字典
        """
        # 准备数据
        X_train, X_test, y_train, y_test, feature_columns = self.prepare_data(data, target_column)

        # 存储结果
        self.results = {
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'target_column': target_column,
            'feature_columns': feature_columns,
            'models': {},
            'best_model': None,
            'comparison': {}
        }

        best_score = float('inf')

        # 训练每个启用的模型
        for model_name, model_config in self.config['models'].items():
            if not model_config.get('enabled', True):
                self.logger.info(f"跳过模型: {model_name} (已禁用)")
                continue

            try:
                self.logger.info(f"训练模型: {model_name}")
                model_result = self._train_single_model(
                    model_name, model_config, X_train, X_test, y_train, y_test
                )

                self.results['models'][model_name] = model_result
                self.models[model_name] = model_result['best_estimator']

                # 更新最佳模型
                test_score = model_result['test_metrics']['rmse']
                if test_score < best_score:
                    best_score = test_score
                    self.best_model = model_result['best_estimator']
                    self.best_model_name = model_name
                    self.results['best_model'] = {
                        'name': model_name,
                        'test_rmse': test_score,
                        'test_r2': model_result['test_metrics']['r2']
                    }

                self.logger.info(f"模型 {model_name} 训练完成, 测试集RMSE: {test_score:.4f}")

            except Exception as e:
                self.logger.error(f"训练模型 {model_name} 失败: {str(e)}")
                continue

        # 生成模型比较
        self._generate_model_comparison()

        return self.results

    def _simplify_cv_results(self, cv_results: Dict) -> Dict:
        """简化交叉验证结果，只保留重要信息"""
        simplified = {}
        important_keys = ['mean_test_score', 'std_test_score', 'params']

        for key in important_keys:
            if key in cv_results:
                simplified[key] = cv_results[key]
        return simplified

    def _train_single_model(self, model_name: str, model_config: Dict[str, Any],
                            X_train: pd.DataFrame, X_test: pd.DataFrame,
                            y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """训练单个模型"""

        # 使用 model_name 记录日志
        self.logger.info(f"开始训练模型: {model_name}")

        try:
            # 实例化模型
            model = self._instantiate_model(model_config)

            # 创建管道（包含标准化）
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])

            # 获取参数网格
            param_grid = model_config.get('param_grid', {})

            # 调整参数名（因为现在在管道中）
            param_grid_pipeline = {}
            for param, values in param_grid.items():
                param_grid_pipeline[f'model__{param}'] = values

            # 交叉验证设置
            cv_folds = model_config.get('cv_folds', 5)
            cv = self._get_cv_strategy(cv_folds)
            scoring = model_config.get('scoring', 'neg_mean_squared_error')
            n_jobs = self.config['training'].get('n_jobs', -1)

            # 超参数调优
            self.logger.info(f"模型 {model_name}: 开始超参数搜索，参数组合数: {len(list(ParameterGrid(param_grid_pipeline)))}")
            grid_search = GridSearchCV(
                pipeline, param_grid_pipeline,
                cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=0
            )

            # 训练模型
            grid_search.fit(X_train, y_train)

            # 获取最佳模型
            best_estimator = grid_search.best_estimator_

            # 预测
            y_train_pred = best_estimator.predict(X_train)
            y_test_pred = best_estimator.predict(X_test)

            # 计算指标
            train_metrics = self._calculate_metrics(y_train, y_train_pred, "train")
            test_metrics = self._calculate_metrics(y_test, y_test_pred, "test")

            # 特征重要性（如果可用）
            feature_importance = self._get_feature_importance(best_estimator, X_train.columns, model_name)

            # 记录训练结果
            self.logger.info(f"模型 {model_name} 训练完成")
            self.logger.info(f"  最佳参数: {grid_search.best_params_}")
            self.logger.info(f"  训练集RMSE: {train_metrics.get('rmse', 'N/A'):.4f}")
            self.logger.info(f"  测试集RMSE: {test_metrics.get('rmse', 'N/A'):.4f}")
            self.logger.info(f"  测试集R²: {test_metrics.get('r2', 'N/A'):.4f}")

            return {
                'model_name': model_name,  # 在结果中包含模型名称
                'best_estimator': best_estimator,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': self._simplify_cv_results(grid_search.cv_results_),
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'feature_importance': feature_importance,
                'predictions': {
                    'train': y_train_pred,
                    'test': y_test_pred
                }
            }

        except Exception as e:
            self.logger.error(f"训练模型 {model_name} 失败: {str(e)}")
            raise



    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, dataset_type: str) -> Dict[str, float]:
        """计算评估指标"""
        metrics = {}

        if 'rmse' in self.config.get('metrics', []):
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))

        if 'mae' in self.config.get('metrics', []):
            metrics['mae'] = mean_absolute_error(y_true, y_pred)

        if 'r2' in self.config.get('metrics', []):
            metrics['r2'] = r2_score(y_true, y_pred)

        if 'mape' in self.config.get('metrics', []):
            metrics['mape'] = self._mean_absolute_percentage_error(y_true, y_pred)

        if 'explained_variance' in self.config.get('metrics', []):
            metrics['explained_variance'] = explained_variance_score(y_true, y_pred)

        return metrics

    def _mean_absolute_percentage_error(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """计算平均绝对百分比误差"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        non_zero_mask = y_true != 0
        if np.sum(non_zero_mask) == 0:
            return np.nan
        return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

    def _get_feature_importance(self, estimator: Pipeline, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """获取特征重要性（如果模型支持）"""
        try:
            # 从管道中获取实际的模型
            if hasattr(estimator, 'named_steps') and 'model' in estimator.named_steps:
                model = estimator.named_steps['model']
            else:
                model = estimator

            # 检查模型是否有特征重要性
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
                feature_importance = dict(zip(feature_names, importance_scores))
                return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

            # 检查线性模型的系数
            elif hasattr(model, 'coef_'):
                coef_scores = model.coef_
                if len(coef_scores.shape) > 1:  # 多输出情况
                    coef_scores = coef_scores[0]
                feature_importance = dict(zip(feature_names, np.abs(coef_scores)))
                return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

        except Exception as e:
            self.logger.warning(f"无法获取特征重要性: {str(e)}")

        return None

    def _generate_model_comparison(self):
        """生成模型比较结果"""
        comparison_data = []

        for model_name, result in self.results['models'].items():
            row = {
                'model': model_name,
                'train_rmse': result['train_metrics'].get('rmse', np.nan),
                'test_rmse': result['test_metrics'].get('rmse', np.nan),
                'train_r2': result['train_metrics'].get('r2', np.nan),
                'test_r2': result['test_metrics'].get('r2', np.nan),
                'train_mae': result['train_metrics'].get('mae', np.nan),
                'test_mae': result['test_metrics'].get('mae', np.nan),
                'best_cv_score': result['best_score']
            }
            comparison_data.append(row)

        self.results['comparison'] = pd.DataFrame(comparison_data)

    def plot_model_comparison(self, metric: str = 'test_rmse', figsize: tuple = (12, 8)):
        """绘制模型比较图"""
        if 'comparison' not in self.results or self.results['comparison'].empty:
            self.logger.warning("没有可用的比较数据")
            return

        df = self.results['comparison']

        plt.figure(figsize=figsize)

        # 根据指标排序
        df_sorted = df.sort_values(metric)

        models = df_sorted['model']
        scores = df_sorted[metric]

        bars = plt.barh(range(len(models)), scores)
        plt.yticks(range(len(models)), models)
        plt.xlabel(metric.upper())
        plt.title(f'模型比较 - {metric.upper()}')

        # 在条形上添加数值
        for i, bar in enumerate(bars):
            plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                     f'{scores.iloc[i]:.4f}',
                     ha='left', va='center', fontweight='bold')

        plt.tight_layout()

        # 保存图片
        os.makedirs('reports/figures', exist_ok=True)
        plt.savefig('reports/figures/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info("模型比较图已保存")

    def plot_feature_importance(self, model_name: str = None, top_n: int = 15, figsize: tuple = (12, 8)):
        """绘制特征重要性图"""
        if model_name is None:
            model_name = self.best_model_name

        if model_name not in self.results['models']:
            self.logger.error(f"模型 {model_name} 不存在")
            return

        feature_importance = self.results['models'][model_name]['feature_importance']
        if feature_importance is None:
            self.logger.warning(f"模型 {model_name} 没有可用的特征重要性")
            return

        # 选择前N个最重要的特征
        top_features = dict(list(feature_importance.items())[:top_n])

        plt.figure(figsize=figsize)
        features, importances = zip(*top_features.items())

        y_pos = np.arange(len(features))
        plt.barh(y_pos, importances, align='center', alpha=0.8)
        plt.yticks(y_pos, features)
        plt.xlabel('特征重要性')
        plt.title(f'{model_name} - Top {top_n} 特征重要性')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        # 保存图片
        os.makedirs('reports/figures', exist_ok=True)
        plt.savefig(f'reports/figures/{model_name}_feature_importance.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"{model_name} 特征重要性图已保存")

    def predict(self, X: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """
        使用训练好的模型进行预测

        Args:
            X: 特征数据
            model_name: 模型名称，如果为None则使用最佳模型

        Returns:
            预测结果
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("没有可用的模型，请先训练模型")
            model = self.best_model
        else:
            if model_name not in self.models:
                raise ValueError(f"模型 {model_name} 不存在")
            model = self.models[model_name]

        return model.predict(X)

    def save_results(self, filepath: str = None):
        """保存训练结果"""
        if filepath is None:
            filepath = f"reports/training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # 转换不可JSON序列化的对象
        def convert_for_json(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif hasattr(obj, '__dict__'):
                return str(obj)  # 对于复杂对象，转换为字符串
            else:
                return obj

        serializable_results = {}
        for key, value in self.results.items():
            if key == 'models':
                serializable_results[key] = {}
                for model_name, model_result in value.items():
                    serializable_results[key][model_name] = {}
                    for sub_key, sub_value in model_result.items():
                        if sub_key == 'best_estimator':
                            # 保存模型到单独的文件
                            model_path = f"models/{model_name}_model.joblib"
                            os.makedirs('models', exist_ok=True)
                            joblib.dump(sub_value, model_path)
                            serializable_results[key][model_name][sub_key] = model_path
                        elif sub_key == 'cv_results':
                            # 简化cv_results，只保留重要信息
                            cv_simple = {
                                'mean_test_score': convert_for_json(sub_value.get('mean_test_score')),
                                'std_test_score': convert_for_json(sub_value.get('std_test_score')),
                                'params': convert_for_json(sub_value.get('params'))
                            }
                            serializable_results[key][model_name][sub_key] = cv_simple
                        else:
                            serializable_results[key][model_name][sub_key] = convert_for_json(sub_value)
            else:
                serializable_results[key] = convert_for_json(value)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"训练结果已保存到: {filepath}")

    def get_summary(self) -> pd.DataFrame:
        """获取训练摘要"""
        if 'comparison' not in self.results:
            return pd.DataFrame()

        return self.results['comparison']