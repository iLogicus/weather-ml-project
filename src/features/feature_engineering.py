# ================================================================
# FEATURE ENGINEERING
# Tạo features từ dữ liệu raw
# ================================================================

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CITIES, PROCESSED_DATA_FILE
from utils.helpers import calculate_daylight_hours


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm các features liên quan đến thời gian.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame gốc
    
    Returns:
    --------
    pd.DataFrame
        DataFrame với temporal features
    """
    df = df.copy()
    
    # Extract từ Date
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Quarter"] = df["Date"].dt.quarter
    
    print("Đã thêm temporal features: Year, Month, Day, DayOfYear, Quarter")
    
    return df


def add_daylight_hours(df: pd.DataFrame, cities: dict = CITIES) -> pd.DataFrame:
    """
    Tính và thêm DaylightHours dựa trên vĩ độ và ngày trong năm.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame có columns: City, DayOfYear
    cities : dict
        Dictionary chứa thông tin cities với latitude
    
    Returns:
    --------
    pd.DataFrame
        DataFrame với DaylightHours column
    """
    df = df.copy()
    
    def get_daylight(row):
        city = row["City"]
        day_of_year = row["DayOfYear"]
        latitude = cities[city]["lat"]
        return calculate_daylight_hours(latitude, day_of_year)
    
    df["DaylightHours"] = df.apply(get_daylight, axis=1)
    
    print("Đã thêm DaylightHours feature")
    
    return df


def add_lag_features(df: pd.DataFrame, lag_days: list = [1, 3]) -> pd.DataFrame:
    """
    Thêm lag features cho nhiệt độ.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame có columns: City, Date, TempMean
    lag_days : list
        Danh sách số ngày lag
    
    Returns:
    --------
    pd.DataFrame
        DataFrame với lag features
    """
    df = df.copy()
    df = df.sort_values(["City", "Date"]).reset_index(drop=True)
    
    for lag in lag_days:
        col_name = f"TempLag{lag}"
        df[col_name] = df.groupby("City")["TempMean"].shift(lag)
    
    # Fill NaN với mean của từng city
    for lag in lag_days:
        col_name = f"TempLag{lag}"
        df[col_name] = df.groupby("City")[col_name].transform(
            lambda x: x.fillna(x.mean())
        )
    
    print(f"Đã thêm lag features: {lag_days}")
    
    return df


def encode_city(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode City thành numerical values (alphabetical order).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame có City column
    
    Returns:
    --------
    pd.DataFrame
        DataFrame với City_encoded column
    """
    df = df.copy()
    
    # Sort cities alphabetically
    city_mapping = {city: idx for idx, city in enumerate(sorted(df["City"].unique()))}
    df["City_encoded"] = df["City"].map(city_mapping)
    
    print(f"Đã encode City: {city_mapping}")
    
    return df


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo tất cả features cần thiết.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame raw
    
    Returns:
    --------
    pd.DataFrame
        DataFrame với tất cả features
    """
    print("\nBẮT ĐẦU FEATURE ENGINEERING...")
    print("="*70)
    
    # 1. Temporal features
    df = add_temporal_features(df)
    
    # 2. Daylight hours
    df = add_daylight_hours(df)
    
    # 3. Lag features
    df = add_lag_features(df, lag_days=[1, 3])
    
    # 4. Encode city
    df = encode_city(df)
    
    # Drop rows with NaN (từ lag features)
    initial_rows = len(df)
    df = df.dropna()
    dropped = initial_rows - len(df)
    
    if dropped > 0:
        print(f"Đã xóa {dropped} rows có NaN")
    
    print("="*70)
    print(f"HOÀN THÀNH: {len(df)} rows với {len(df.columns)} columns")
    
    return df


def get_feature_importance_info(df: pd.DataFrame) -> dict:
    """
    Lấy thông tin về features để phân tích.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame đã có features
    
    Returns:
    --------
    dict
        Dictionary chứa thông tin features
    """
    feature_info = {
        "temporal": ["Year", "Month", "Day", "DayOfYear", "Quarter"],
        "weather": ["Humidity", "Precipitation", "WindSpeed"],
        "temperature": ["TempMax", "TempMean", "TempMin"],
        "engineered": ["DaylightHours", "TempLag1", "TempLag3", "City_encoded"],
        "all_features": df.columns.tolist()
    }
    
    return feature_info


def calculate_lag_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính trung bình lag features theo City và Month để dùng cho prediction.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame có TempLag1, TempLag3
    
    Returns:
    --------
    pd.DataFrame
        DataFrame với index (City, Month) và columns là lag means
    """
    lag_stats = df.groupby(["City", "Month"]).agg({
        "TempLag1": "mean",
        "TempLag3": "mean"
    }).round(2)
    
    print(f"Đã tính lag statistics cho {len(lag_stats)} City-Month combinations")
    
    return lag_stats


def save_processed_data(df: pd.DataFrame, filepath: str = PROCESSED_DATA_FILE) -> None:
    """
    Lưu processed data vào CSV.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame đã processed
    filepath : str
        Đường dẫn file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Đã lưu processed data: {filepath}")


def load_processed_data(filepath: str = PROCESSED_DATA_FILE) -> pd.DataFrame:
    """
    Load processed data từ CSV.
    
    Parameters:
    -----------
    filepath : str
        Đường dẫn file
    
    Returns:
    --------
    pd.DataFrame
        DataFrame đã processed
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")
    
    df = pd.read_csv(filepath, parse_dates=["Date"])
    print(f"Loaded processed data: {len(df)} rows")
    
    return df


if __name__ == "__main__":
    # Test module
    print("Testing feature_engineering module...")
    
    from data.load_data import load_raw_data, validate_data
    
    # Load raw data
    df_raw = load_raw_data(fetch_new=False)
    df_raw = validate_data(df_raw)
    
    # Create features
    df_processed = create_all_features(df_raw)
    
    # Calculate lag stats
    lag_stats = calculate_lag_statistics(df_processed)
    print(f"\nLag statistics shape: {lag_stats.shape}")
    print(lag_stats.head())
    
    # Save
    save_processed_data(df_processed)
    
    # Feature info
    feature_info = get_feature_importance_info(df_processed)
    print(f"\nFeatures by category:")
    for category, features in feature_info.items():
        if category != "all_features":
            print(f"   {category}: {len(features)} features")
