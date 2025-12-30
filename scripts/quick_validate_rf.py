import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import joblib

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 设置matplotlib后端
import matplotlib
matplotlib.use('Agg')
from src.model.model_trainer import ModelTrainer

modeltrainer = ModelTrainer()

train_file_path = 'data/processed/features_engineered_data.csv'
test_file_path = 'data/processed/features_engineered_test_data.csv'
train_data = pd.read_csv(train_file_path, sep=',')
test_data = pd.read_csv(test_file_path, sep=',')
core_weather_features = ['global_rad:W',       # 全球辐射 (最重要！)
                             'clear_sky_rad:W',    # 晴空辐射
                             'diffuse_rad:W',      # 散射辐射
                             'direct_rad:W',       # 直接辐射
                             'temp',               # 温度
                             'relative_humidity_2m:p', # 湿度
                             'total_cloud_cover:p',    # 云量
                             'sun_elevation:d',
                             'is_daylight',
                             'pv_change_1h',
                             'global_rad_acceleration_1h']
time_features = ['hour', 'day_of_year', 'day_of_week', 'month']
selected_columns = core_weather_features + time_features

X_train, X_test, y_train, y_test, selected_columns =modeltrainer.prepare_data(train_data, test_data, selected_columns, 'pv_production')
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def quick_validate_rf(X_train, X_test, y_train, y_test):
    """快速验证随机森林模型"""

    print("训练随机森林模型...")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 使用默认参数的随机森林
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1  # 使用所有CPU核心加速
    )

    rf_model.fit(X_train_scaled, y_train)

    # 6. 快速预测和评估
    print(" 模型评估...")
    y_train_pred = rf_model.predict(X_train_scaled)
    y_test_pred = rf_model.predict(X_test_scaled)

    # 计算关键指标
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # 7. 输出结果
    print("\n" + "="*50)
    print("快速验证结果")
    print("="*50)
    print(f" 模型: 随机森林 (n_estimators=100)")
    print(f"📁 数据: {X_train.shape[1]} 个特征, {X_train.shape[0]} 个样本")
    print(f" 目标: PV_production")
    print("\n 性能指标:")
    print(f"   训练集 RMSE: {train_rmse:.4f}")
    print(f"   测试集 RMSE: {test_rmse:.4f}")
    print(f"   训练集 R²:  {train_r2:.4f}")
    print(f"   测试集 R²:  {test_r2:.4f}")
    print(f"   训练集 MAE: {train_mae:.4f}")
    print(f"   测试集 MAE: {test_mae:.4f}")

    # 8. 解释R²分数
    print(f"\n💡 R²分数解释:")
    if test_r2 > 0.9:
        interpretation = "优秀 - 模型解释了大量方差"
    elif test_r2 > 0.7:
        interpretation = "良好 - 模型表现不错"
    elif test_r2 > 0.5:
        interpretation = "一般 - 模型有改进空间"
    elif test_r2 > 0.3:
        interpretation = "较差 - 需要显著改进"
    else:
        interpretation = "很差 - 考虑重新设计特征或模型"

    print(f"   {interpretation}")


    return True, {
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'model': rf_model
    }


if __name__ == "__main__":
    modeltrainer = ModelTrainer()

    train_file_path = 'data/processed/features_engineered_data.csv'
    test_file_path = 'data/processed/features_engineered_test_data.csv'
    train_data = pd.read_csv(train_file_path, sep=',')
    test_data = pd.read_csv(test_file_path, sep=',')
    core_weather_features = ['global_rad:W',       # 全球辐射 (最重要！)
                             'clear_sky_rad:W',    # 晴空辐射
                             'diffuse_rad:W',      # 散射辐射
                             'direct_rad:W',       # 直接辐射
                             'temp',               # 温度
                             'relative_humidity_2m:p', # 湿度
                             'total_cloud_cover:p',    # 云量
                             'sun_elevation:d',
                             'is_daylight',
                             'pv_change_1h',
                             'global_rad_acceleration_1h']
    time_features = ['hour', 'day_of_year', 'day_of_week', 'month']
    selected_columns = core_weather_features + time_features

    X_train, X_test, y_train, y_test, selected_columns =modeltrainer.prepare_data(train_data, test_data, selected_columns, 'pv_production')

    success, results = quick_validate_rf(X_train, X_test, y_train,y_test)