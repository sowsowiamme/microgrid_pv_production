import sys
import os
import pandas as pd
import numpy as np
import logging

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.features.feature_selection import FeatureSelector

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_pv_production_feature_selection():
    """验证特征筛选功能"""

    print("\n开始验证特征筛选...")

    try:
        # 1. 加载之前处理的特征工程数据
        processed_feature_data_path = 'data/processed/features_engineered_data.csv'
        if not os.path.exists(processed_feature_data_path):
            raise FileNotFoundError(f"未找到特征工程数据文件: {processed_feature_data_path}。请先运行特征工程验证。")

        engineered_data = pd.read_csv(processed_feature_data_path)

        print(f" 加载特征工程数据成功，形状: {engineered_data.shape}")
        print(f"   特征数量: {len(engineered_data.columns)}")
        print(f"   特征列表: {list(engineered_data.columns)}")

        target_column = 'pv_production'

        # 3. 初始化特征选择器
        feature_selector = FeatureSelector(
            target_column=target_column,
            correlation_threshold=0.1,  # 可以调整这个阈值
            top_k_features=25           # 选择前25个最重要的特征
        )

        print(" FeatureSelector 初始化成功")

        # 4. 执行相关性分析
        print("\n 执行相关性分析...")
        high_corr_features = feature_selector.correlation_analysis(engineered_data, plot=True)

        print(f"高相关性特征 (>= {feature_selector.correlation_threshold}):")
        for feature, corr in high_corr_features.items():
            print(f"   {feature}: {corr:.3f}")
        print(f"\n 特征筛选验证成功!")
        return True, high_corr_features

    except Exception as e:
        print(f"❌ 特征筛选验证失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    success, selected_data = validate_pv_production_feature_selection()

    # 高相关性特征 (>= 0.1):
    #    global_rad:W: 0.864
    #    global_rad_1h:Wh: 0.845
    #    direct_rad:W: 0.841
    #    direct_rad_1h:Wh: 0.824
    #    clear_sky_rad:W: 0.766
    #    clear_sky_energy_1h:J: 0.751
    #    diffuse_rad:W: 0.658
    #    diffuse_rad_1h:Wh: 0.646
    #    sun_elevation:d: 0.642
    #    pv_change_1h: 0.554
    #    sunshine_duration_1h:min: 0.549
    #    is_daylight: 0.541
    #    temp: 0.434
    #    t_10m:C: 0.430
    #    t_50m:C: 0.411
    #    t_100m:C: 0.396
    #    dew_point_2m:C: 0.217
    #    dew_point_10m:C: 0.212
    #    global_rad_trend_1h: 0.197
    #    dew_point_50m:C: 0.195
    #    dew_point_100m:C: 0.187
    #    wind_speed_50m:ms: -0.101
    #    wind_dir_50m:d: -0.104
    #    wind_dir_100m:d: -0.108
    #    high_cloud_cover:p: -0.125
    #    low_cloud_cover:p: -0.134
    #    spot_market_price: -0.157
    #    consumption: -0.159
    #    wind_speed_100m:ms: -0.163
    #    total_cloud_cover:p: -0.192
    #    medium_cloud_cover:p: -0.209
    #    effective_cloud_cover:p: -0.223
    #    global_rad_acceleration_1h: -0.250
    #    relative_humidity_100m:p: -0.455
    #    relative_humidity_50m:p: -0.497
    #    relative_humidity_10m:p: -0.540
    #    relative_humidity_2m:p: -0.549
