from importlib import import_module
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
import os
import statsmodels.stats.multitest as smm
import scipy.stats as stats

# from src.model.compare_and_tune_models import model_name
warnings.filterwarnings('ignore')
from sklearn.metrics import make_scorer, mean_squared_error

# 机器学习相关
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import (train_test_split, TimeSeriesSplit, KFold,
                                     cross_val_score, GridSearchCV, RandomizedSearchCV,
                                     ParameterGrid)
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
except Exception as exc:
    XGB_AVAILABLE = False
    logging.warning("XGBoost not available: %s", exc)

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except Exception as exc:
    LGBM_AVAILABLE = False
    logging.warning("LightGBM not available: %s", exc)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

        raw_config = self._load_config(config_path)
        self.config = raw_config
        self.context = self._build_context()
        self.config = self._resolve_config(raw_config)
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        # self.scaler = StandardScaler()

        # # 获取CV配置
        training_config = self.config.get('training', {})
        self.cv_strategy = training_config.get('cv_strategy', 'timeseries')
        self.cv_splits = int(training_config.get('cv_splits', 5))
        self.random_state = int(training_config.get('random_state', self.context.get('random_state', 42)))
        self.n_jobs = int(training_config.get('n_jobs', -1))
        self.metrics = training_config.get('metrics', ['rmse', 'mae', 'r2', 'mape'])

        # 获取CV分割器
        self.cv = self._get_cv_strategy()

        self.logger.info(
        f"ModelTrainer 初始化完成 | "
        f"CV策略: {self.cv_strategy} | "
        f"折数: {self.cv_splits} | "
        f"随机种子: {self.random_state}"
        )



    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载模型配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
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
                'cv_splits': 5,
                'metrics': ['rmse', 'mae', 'r2', 'mape'],
                'scoring': 'neg_mean_squared_error',
                'n_jobs': -1,
            }
        }

    def prepare_data(
        self,
        train_data: pd.DataFrame,
        test_data: Optional[pd.DataFrame] = None,
        selected_columns: Optional[List[str]] = None,
        target_column: str = 'pv_production',
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
        """
        准备训练数据。

        当前项目已经在数据层按时间拆好了 train/test，因此这里默认接收两份
        DataFrame，避免在训练器内部再次随机切分导致时间泄漏。

        Args:
            train_data: 训练集
            test_data: 测试集；如果不传，则按时间顺序从 train_data 尾部切出测试集
            selected_columns: 显式选择的特征列；不传时使用除目标列外的数值列
            target_column: 目标变量列名

        Returns:
            (X_train, X_test, y_train, y_test, feature_columns)
        """
        if target_column not in train_data.columns:
            raise ValueError(f"目标列 '{target_column}' 不存在于训练集中")

        if test_data is None:
            test_size = float(self.config.get('training', {}).get('test_size', 0.2))
            split_idx = int(len(train_data) * (1 - test_size))
            test_data = train_data.iloc[split_idx:].copy()
            train_data = train_data.iloc[:split_idx].copy()
            self.logger.info(f"未传入测试集，按时间顺序切分: split_idx={split_idx}, test_size={test_size}")

        if target_column not in test_data.columns:
            raise ValueError(f"目标列 '{target_column}' 不存在于测试集中")

        if selected_columns is None:
            numeric_columns = train_data.select_dtypes(include=[np.number]).columns.tolist()
            selected_columns = [col for col in numeric_columns if col != target_column]

        missing_train = [col for col in selected_columns if col not in train_data.columns]
        missing_test = [col for col in selected_columns if col not in test_data.columns]
        if missing_train or missing_test:
            raise ValueError(
                f"特征列缺失: train_missing={missing_train}, test_missing={missing_test}"
            )

        feature_columns = list(selected_columns)
        X_train = train_data[feature_columns].copy()
        y_train = train_data[target_column].copy()
        X_test = test_data[feature_columns].copy()
        y_test = test_data[target_column].copy()

        self.logger.info(f"数据准备完成: 训练集={X_train.shape}, 测试集={X_test.shape}")
        return X_train, X_test, y_train, y_test, feature_columns

    def _get_cv_strategy(self, cv_splits: Optional[int] = None):
        """获取配置的交叉验证策略"""
        n_splits = int(cv_splits or self.cv_splits)
        strategy = str(self.cv_strategy).lower()
        if strategy == 'timeseries':
            logger.info(f"使用时间序列交叉验证 (n_splits={n_splits})")
            return TimeSeriesSplit(n_splits=n_splits)
        elif strategy == 'kfold':
            logger.info(
                f"使用K-Fold交叉验证 (n_splits={n_splits}, "
                f"random_state={self.random_state})"
            )
            return KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state
            )
        else:
            raise ValueError(
                f"不支持的CV策略: {self.cv_strategy}。"
                "请选择 'timeseries' 或 'kfold'"
            )

    def train_models(
        self,
        train_data: pd.DataFrame,
        test_data: Optional[pd.DataFrame] = None,
        selected_columns: Optional[List[str]] = None,
        target_column: str = 'pv_production',
    ) -> Dict[str, Any]:
        """
        训练所有启用的模型

        Args:
            train_data: 训练数据
            test_data: 测试数据；不传时按时间顺序从 train_data 尾部切分
            selected_columns: 特征列
            target_column: 目标变量列名

        Returns:
            训练结果字典
        """
        # 准备数据
        X_train, X_test, y_train, y_test, feature_columns = self.prepare_data(
            train_data=train_data,
            test_data=test_data,
            selected_columns=selected_columns,
            target_column=target_column,
        )

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
            param_grid = self._resolve_config(model_config.get('param_grid', {}))

            # 调整参数名（因为现在在管道中）
            param_grid_pipeline = {}
            for param, values in param_grid.items():
                param_grid_pipeline[f'model__{param}'] = values

            # 交叉验证设置
            cv_splits = model_config.get('cv_folds', self.cv_splits)
            cv = self._get_cv_strategy(cv_splits)
            training_config = self.config.get('training', {})
            scoring = training_config.get('scoring', 'neg_mean_squared_error')
            n_jobs = training_config.get('n_jobs', self.n_jobs)
            grid_verbose = int(training_config.get('grid_search_verbose', 0))

            # 超参数调优
            self.logger.info(f"模型 {model_name}: 开始超参数搜索，参数组合数: {len(list(ParameterGrid(param_grid_pipeline)))}")
            grid_search = GridSearchCV(
                pipeline, param_grid_pipeline,
                cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=grid_verbose
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
            feature_importance = self._get_feature_importance(best_estimator, list(X_train.columns))

            # 记录训练结果
            self.logger.info(f"模型 {model_name} 训练完成")
            self.logger.info(f"  最佳参数: {grid_search.best_params_}")
            self.logger.info(f"  训练集RMSE: {train_metrics.get('rmse', 'N/A'):.4f}")
            self.logger.info(f"  测试集RMSE: {test_metrics.get('rmse', 'N/A'):.4f}")
            self.logger.info(f"  测试集R²: {test_metrics.get('r2', 'N/A'):.4f}")

            return {
                'model_name': model_name,  # 在结果中包含模型名称
                'best_estimator': best_estimator,
                'best_params': {k.replace('model__', ''): v for k, v in grid_search.best_params_.items()},
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
        configured_metrics = self.metrics

        if 'rmse' in configured_metrics:
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))

        if 'mae' in configured_metrics:
            metrics['mae'] = mean_absolute_error(y_true, y_pred)

        if 'r2' in configured_metrics:
            metrics['r2'] = r2_score(y_true, y_pred)

        if 'mape' in configured_metrics:
            metrics['mape'] = self._mean_absolute_percentage_error(y_true, y_pred)

        if 'explained_variance' in configured_metrics:
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

    def save_inference_bundle(
        self,
        results: Dict[str, Any],
        comparison_df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        path: Optional[str] = None,
    ) -> str:
        """
        将 Mean CV RMSE 排名第一的 Pipeline（含 StandardScaler）与特征元数据保存为 joblib，
        供 Flask 等推理服务加载。
        """
        if path is None:
            root = Path(__file__).resolve().parents[2]
            path = str(root / "models" / "pv_inference_bundle.joblib")
        model_col = "Model" if "Model" in comparison_df.columns else "model"
        best_name = comparison_df.iloc[0][model_col]
        best_result = results[best_name]
        pipeline = best_result.get("best_model") or best_result.get("best_estimator")
        if pipeline is None:
            raise ValueError(f"结果中找不到可保存的模型: {best_name}")
        bundle = {
            "pipeline": pipeline,
            "feature_columns": list(feature_columns),
            "target_column": target_column,
            "model_name": best_name,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, path)
        self.logger.info(f"推理模型包已保存: {path} (模型: {best_name})")
        return path

    def tune_and_evaluate_model(self,
        model: BaseEstimator,
        param_grid: Dict[str, Any],
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        时间序列安全的模型调优与评估。

        1. 使用配置的CV策略进行超参数搜索
        2. 在测试集上评估
        3. 收集CV得分用于显著性检验
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 开始训练和调优: {model_name}")
        logger.info(f"{'='*60}")

        try:
            # 创建标准化管道
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])

            # 调整参数名以适应管道
            param_grid_pipeline = {f'model__{k}': v for k, v in param_grid.items()}

            # 获取训练配置
            training_config = self.config.get('training', {})
            scoring = training_config.get('scoring', 'neg_mean_squared_error')
            n_jobs = self.n_jobs
            grid_verbose = int(training_config.get('grid_search_verbose', 1))

            n_combinations = len(ParameterGrid(param_grid_pipeline))
            n_cv_folds = getattr(self.cv, 'n_splits', self.cv_splits)
            logger.info(
                f"  开始超参数搜索... 超参维度 {len(param_grid_pipeline)} 个, "
                f"网格组合数 {n_combinations}, CV 折数 {n_cv_folds} "
                f"(约 {n_combinations * n_cv_folds} 次带 CV 的拟合; 大数据集可能需数分钟至更久)"
            )
            if grid_verbose == 0:
                logger.info(
                    "  (当前 grid_search_verbose=0 无过程输出; "
                    "可在 config/model_config.yaml 的 training 下设置 grid_search_verbose: 1)"
                )

            grid = GridSearchCV(
                pipeline,
                param_grid_pipeline,
                cv=self.cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=grid_verbose,
            )
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            best_params = {k.replace('model__', ''): v for k, v in grid.best_params_.items()}

            # 测试集评估
            y_pred = best_model.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_mae = mean_absolute_error(y_test, y_pred)
            test_r2 = r2_score(y_test, y_pred)

            # 获取CV得分（用于显著性检验）
            cv_scores = cross_val_score(
                best_model,
                X_train,
                y_train,
                cv=self.cv,
                scoring=make_scorer(mean_squared_error, greater_is_better=False)
            )
            cv_rmses = np.sqrt(-cv_scores)

            # 计算指标
            train_pred = best_model.predict(X_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            train_mae = mean_absolute_error(y_train, train_pred)
            train_r2 = r2_score(y_train, train_pred)

            logger.info(f"  ✅ 训练完成!")
            logger.info(f"    最佳参数: {best_params}")
            logger.info(f"    测试集 R²: {test_r2:.4f}")
            logger.info(f"    训练集 RMSE: {train_rmse:.4f}")
            logger.info(f"    训练集 R²: {train_r2:.4f}")
            logger.info(f"    测试集 RMSE: {test_rmse:.4f}")
            logger.info(f"    平均 CV RMSE: {np.mean(cv_rmses):.4f} ± {np.std(cv_rmses):.4f}")

            return {
                'best_model': best_model,
                'best_params': best_params,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'train_mae': train_mae,
                'train_r2': train_r2,
                'cv_rmses': cv_rmses,
                'mean_cv_rmse': np.mean(cv_rmses),
                'std_cv_rmse': np.std(cv_rmses),
                'y_pred': y_pred,
                'y_true': y_test
            }

        except Exception as e:
            logger.error(f"❌ {model_name} 训练失败: {str(e)}")
            raise

    def _build_context(self) -> Dict[str, Any]:
        """构建变量上下文（用于替换 ${xxx}）"""
        import multiprocessing
        n_jobs_config = self.config.get('global', {}).get('n_jobs', -1)
        n_jobs = multiprocessing.cpu_count() if n_jobs_config == -1 else n_jobs_config
        return {
            'n_jobs': n_jobs,
            'random_state': self.config.get('global', {}).get('random_state', 42)
        }
    def _resolve_value(self, value: Any) -> Any:
        """解析变量（支持${xxx}）"""
        if isinstance(value, str):
            if value.startswith('${') and value.endswith('}'):
                key = value[2:-1]
                if key in self.context:
                    return self.context[key]
        return value


    def _resolve_config(self, obj: Any) -> Any:
        """递归解析配置中的占位符"""
        if isinstance(obj, dict):
            return {k: self._resolve_config(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._resolve_config(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            key = obj[2:-1].strip()
            if key in self.context:
                return self.context[key]
            else:
                logger.warning(f"⚠️  未定义的变量: ${key}，保留原值")
                return obj  # 保留原字符串但警告
        else:
            return obj  # 非字符串/非占位符直接返回

    def _instantiate_model(self, model_config: Dict[str, Any]) -> BaseEstimator:
        resolved_config = self._resolve_config(model_config)
        class_path = resolved_config['class']
        init_params = resolved_config.get('init_params', {})
        try:
            model_path, model_name = class_path.rsplit('.',1)
            module = import_module(model_path)
            ModuleClass = getattr(module, model_name)
            model = ModuleClass(**init_params)
            return model
        except Exception as e:
            logger.error(f"❌ 模型初始化失败 ({class_path}): {str(e)}")
            raise


    def compare_models_with_significance(
        self,
        results: Dict[str, Dict[str, Any]],
        alpha: float = 0.05,
        save_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        模型显著性比较
        1. 生成性能对比表
        2. 执行配对t检验（单侧）
        3. 识别统计显著的差异

        Args:
            results: tune_and_evaluate_model 返回的结果字典
            alpha: 显著性水平 (默认 0.05)
            save_path: 保存结果的路径

        Returns:
            (comparison_df, p_value_matrix)
        """
        logger.info(f"\n{'='*60}")
        logger.info("📊 模型显著性比较")
        logger.info(f"{'='*60}")

        model_names = list(results.keys())
        n_models = len(model_names)

        # 构建性能比较表
        records = []
        cv_rmses_dict = {}

        for name, res in results.items():
            records.append({
                'Model': name,
                'Test RMSE': res['test_rmse'],
                'Train RMSE': res['train_rmse'],
                'Mean CV RMSE': res['mean_cv_rmse'],
                'Std CV RMSE': res['std_cv_rmse']
            })
            cv_rmses_dict[name] = res['cv_rmses']

        comparison_df = pd.DataFrame(records).sort_values('Mean CV RMSE').reset_index(drop=True)

        # 构建p-value矩阵
        p_value_matrix = pd.DataFrame(
            np.ones((n_models, n_models)),
            index=model_names,
            columns=model_names
        )

        # 配对t检验（单侧：我们关心A是否显著优于B）
        logger.info(f"\n🔍 配对t检验 (α={alpha}, 单侧检验):")
        for i, model_a in enumerate(model_names):
            for j, model_b in enumerate(model_names):
                if i >= j:
                    continue

                try:
                    # ttest_rel 返回双侧p值，我们除以2得到单侧
                    t_stat, p_val = stats.ttest_rel(
                        cv_rmses_dict[model_a],
                        cv_rmses_dict[model_b]
                    )
                    p_val_one_sided = p_val / 2 if t_stat < 0 else 1 - (p_val / 2)

                    p_value_matrix.loc[model_a, model_b] = p_val_one_sided
                    p_value_matrix.loc[model_b, model_a] = p_val_one_sided

                    # 格式化输出
                    better = model_a if np.mean(cv_rmses_dict[model_a]) < np.mean(cv_rmses_dict[model_b]) else model_b
                    worse = model_b if better == model_a else model_a

                    if p_val_one_sided < alpha:
                        logger.info(
                            f"✅ {better} 显著优于 {worse} "
                            f"(p={p_val_one_sided:.4f})"
                        )
                    else:
                        logger.info(
                            f"⚠️  {better} 与 {worse} 无显著差异 "
                            f"(p={p_val_one_sided:.4f})"
                        )

                except Exception as e:
                    logger.warning(f"⚠️  {model_a} vs {model_b} 检验失败: {str(e)}")
                    p_value_matrix.loc[model_a, model_b] = 1.0
                    p_value_matrix.loc[model_b, model_a] = 1.0

        # 保存结果
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            comparison_df.to_csv(f"{save_path}_comparison.csv", index=False)
            p_value_matrix.to_csv(f"{save_path}_pvalues.csv")
            logger.info(f"结果已保存到: {save_path}")

        return comparison_df, p_value_matrix


    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        metric: str = 'Mean CV RMSE',
        figsize: Tuple[int, int] = (12, 8),
        save_path: str = 'reports/figures/model_comparison.png'
    ):
        if comparison_df.empty:
            logger.warning("没有可绘制的模型比较数据")
            return
        if metric not in comparison_df.columns:
            raise ValueError(f"比较表中不存在指标列: {metric}")

        # 按指标排序
        df_sorted = comparison_df.sort_values(metric)
        model_col = 'Model' if 'Model' in df_sorted.columns else 'model'
        models = df_sorted[model_col]
        scores = df_sorted[metric]

        # 创建水平条形图
        plt.figure(figsize=figsize)
        bars = plt.barh(range(len(models)), scores, color='skyblue')
        plt.yticks(range(len(models)), models)
        plt.xlabel(metric)
        plt.title(f'Model Comparison - {metric}')
        plt.gca().invert_yaxis()

        # 在条形上添加数值
        for i, bar in enumerate(bars):
            plt.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height()/2,
                f'{scores.iloc[i]:.4f}'),
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 模型比较图已保存到: {save_path}")
