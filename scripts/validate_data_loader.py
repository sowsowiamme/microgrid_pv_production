import sys
import os
import pandas as pd
import logging

# 添加src目录到Python路径
sys.path.append('src')

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 现在可以导入 src 中的模块
from src.data.loaders import DataLoader

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_data_loader():
    """使用真实数据验证DataLoader"""

    # 初始化DataLoader
    data_loader = DataLoader(
        daily_points=24,
        time_column='time',
        target_columns=['pv_production', 'consumption']  # 根据你的数据调整
    )

    try:
        # put your own file path
        file_path = '/Users/sowsow/Documents/2024 job finding/3. Rye微电网数据集/rye-ai-hackathon/data/train.csv'  # 替换为你的真实文件路径

        print("开始验证 DataLoader...")
        print(f"数据文件: {file_path}")

        # 测试数据加载
        print("\n1. 测试数据加载...")
        data = data_loader.load_data(file_path)
        print(f"数据加载成功! 形状: {data.shape}")

        # 检查基本信息
        print("\n2. 数据基本信息:")
        print(f"   列名: {list(data.columns)}")
        print(f"   时间范围: {data.index.min()} 到 {data.index.max()}")
        print(f"   总天数: {data['date'].nunique()}")

        # 检查数据完整性
        print("\n3. 数据完整性检查:")
        print(f"   缺失值数量: {data.isnull().sum().sum()}")
        print(f"   重复行数量: {data.duplicated().sum()}")

        # 检查每日数据点数
        print("\n4. 每日数据分布:")
        daily_counts = data.groupby('date').size()
        print(f"   平均每日数据点: {daily_counts.mean():.2f}")
        print(f"   最小每日数据点: {daily_counts.min()}")
        print(f"   最大每日数据点: {daily_counts.max()}")

        # 显示前几行数据
        print("\n5. 数据预览:")
        print(data.head())

        # 测试特征创建功能
        print("\n6. 测试时间特征创建...")
        data_with_features = data_loader.create_time_features(data)
        new_columns = set(data_with_features.columns) - set(data.columns)
        print(f"✅ 新增特征列: {list(new_columns)}")

        print("\n🎉 DataLoader 验证完成! 所有功能正常!")
        return True

    except Exception as e:
        print(f"验证过程中出错: {str(e)}")
        return False

if __name__ == "__main__":
    validate_data_loader()