import sys
import os
import pandas as pd
import logging


# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.features.feature_engineer import FeaturePreprocessor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_processed_data(test_data= False):
    """
    加载之前 DataLoader 处理过的数据
    可以从文件加载，或者直接从前一个流程传递
    """
    # 方法1: 从保存的文件加载
    if test_data:
        processed_data_path = 'data/processed/loaded_test_data.csv'
    else:
        processed_data_path = 'data/processed/loaded_data.csv'
    if os.path.exists(processed_data_path):
        print(f"📂 从文件加载已处理的数据: {processed_data_path}")
        # with open(processed_data_path, 'csv', sep = '') as f:
        #     return pickle.load(f)
        pd.read_csv(processed_data_path, sep=',')

    # 方法2: 如果没有保存的文件，重新运行 DataLoader
    print("⚠️  未找到已保存的数据文件，重新运行 DataLoader...")
    from scripts.validate_data_loader import validate_data_loader
    success, data = validate_data_loader(test_data)

    if success and data is not None:
        # 保存数据供后续使用
        os.makedirs('data/processed', exist_ok=True)
        # with open(processed_data_path, 'wb') as f:
        #     pickle.dump(data, f)
        data.to_csv(processed_data_path, sep=',')
        print(f"数据已保存到: {processed_data_path}")
        return data
    else:
        raise ValueError("无法加载数据，请先运行 validate_data_loader.py")

def validate_radiation_features_on_real_data(test_data=False):
    """在真实数据上验证辐射特征功能"""

    print("🔍 开始在真实数据上验证辐射特征工程...")

    try:
        # 1. 加载之前处理的数据
        processed_data = load_processed_data(test_data)
        print(f"✅ 加载数据成功，形状: {processed_data.shape}")
        print("数据列:", list(processed_data.columns))

        # 2. 检查必要的列是否存在
        required_columns = ['global_rad:W', 'sun_elevation:d', 'pv_production']
        missing_columns = [col for col in required_columns if col not in processed_data.columns]

        if missing_columns:
            print(f"⚠️  缺少必要的列: {missing_columns}")
            print("当前可用列:", list(processed_data.columns))

            # 尝试使用可用的类似列
            available_radiation_cols = [col for col in processed_data.columns if 'rad' in col.lower()]
            available_elevation_cols = [col for col in processed_data.columns if any(x in col.lower() for x in ['elev', 'sun', 'solar'])]
            available_pv_cols = [col for col in processed_data.columns if any(x in col.lower() for x in ['pv', 'power', 'production'])]

            print(f"可用的辐射相关列: {available_radiation_cols}")
            print(f"可用的高度角相关列: {available_elevation_cols}")
            print(f"可用的光伏相关列: {available_pv_cols}")

            # 使用第一个可用的列作为替代
            radiation_col = available_radiation_cols[0] if available_radiation_cols else None
            elevation_col = available_elevation_cols[0] if available_elevation_cols else None
            pv_col = available_pv_cols[0] if available_pv_cols else None

            if not radiation_col:
                raise ValueError("没有找到可用的辐射数据列")
        else:
            radiation_col = 'global_rad:W'
            elevation_col = 'sun_elevation:d'
            pv_col = 'pv_production'

        # 3. 初始化特征预处理器
        preprocessor = FeaturePreprocessor(
            target_columns=[pv_col] if pv_col else [],
            scaling_method='standard'
        )
        print("✅ FeaturePreprocessor 初始化成功")

        # 4. 测试辐射特征（使用实际数据）
        print("\n 在真实数据上测试辐射特征创建...")

        # 4. 在原有数据上创建lag特征，先仅针对目标列
        # processed_data = preprocessor.create_lag_features(processed_data, ['pv_production'])

        # # 创建时间特征（基础）
        # data_with_time = preprocessor.create_time_features(processed_data)

        # 创建辐射特征（使用检测到的列名）
        data_with_radiation = preprocessor.create_radiation_features(
            processed_data,
            radiation_col=radiation_col,
            pv_production_col=pv_col,
            elevation_col=elevation_col
        )

        # 检查新创建的辐射特征
        new_radiation_features = [col for col in data_with_radiation.columns
                                  if col not in processed_data.columns
                                  and any(x in col for x in ['rad', 'pv', 'daylight', 'light', 'efficiency'])]

        print(f"✅ 创建的辐射特征 ({len(new_radiation_features)} 个):")
        for feature in new_radiation_features:
            print(f"   - {feature}")
        data_after_fill = preprocessor.handle_missing_values(data_with_radiation, strategy="forward_fill")
        #
        # # 5. 测试高级时间特征
        # print("\n⏰ 测试高级时间特征...")
        # data_with_advanced_time = preprocessor.create_advanced_time_features(data_with_radiation)
        #
        # advanced_time_features = [col for col in data_with_advanced_time.columns
        #                           if col not in data_with_radiation.columns
        #                           and any(x in col for x in ['solar', 'seasonal', 'peak', 'sunrise', 'advanced'])]
        #
        # print(f"✅ 创建的高级时间特征 ({len(advanced_time_features)} 个):")
        # for feature in advanced_time_features:
        #     print(f"   - {feature}")
        #
        # # 6. 测试完整流程（无数据泄漏版本）
        # print("\n🔄 测试完整特征工程流程（无数据泄漏）...")
        # complete_features = preprocessor.create_leakage_free_features(processed_data, is_training=True)

        # 生成特征摘要
        print('\n看填充后数据集的数据头长什么样')
        print(data_after_fill.head())
        print('\n把填充后的数据集进行保存，用于后续特征筛选')
        if test_data:
            processed_feature_data_path = 'data/processed/features_engineered_test_data.csv'
        else:
            processed_feature_data_path = 'data/processed/features_engineered_data.csv'
        data_after_fill.to_csv(processed_feature_data_path, sep =',')

        summary = preprocessor.get_feature_summary(data_after_fill)


        print(f"\n📈 完整特征工程结果:")
        print(f"   总特征数: {summary['total_features']}")
        print(f"   数值特征: {summary['numeric_features']}")
        print(f"   缺失值: {summary['missing_values']}")
        print(f"   数据形状: {summary['shape']}")

        # 7. 显示关键特征的统计信息
        print(f"\n🔍 关键特征统计:")
        key_features = []
        if radiation_col:
            key_features.extend(['global_rad_trend_1h', 'global_rad_acceleration_1h', 'radiation_efficiency'])
        if elevation_col:
            key_features.append('is_daylight')
        key_features.extend(['is_peak_solar_hours'])

        for feature in key_features:
            if feature in data_with_radiation.columns:
                stats = data_with_radiation[feature].describe()
                print(f"   {feature}:")
                print(f"      count={stats['count']}, mean={stats['mean']:.3f}, std={stats['std']:.3f}")
                print(f"      min={stats['min']:.3f}, max={stats['max']:.3f}")


        # 8. 保存处理后的数据
        # output_path = 'data/processed/features_engineered_data.pkl'
        # with open(output_path, 'wb') as f:
        #     pickle.dump(complete_features, f)
        # print(f"\n💾 特征工程完成的数据已保存到: {output_path}")
        #
        # print(f"\n 在真实数据上的辐射特征验证成功!")

        return True, summary

    except Exception as e:
        print(f" 验证失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None



if __name__ == "__main__":
    success, final_data = validate_radiation_features_on_real_data(test_data=False)