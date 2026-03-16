# ================================================================
# DATA LOADING & VALIDATION
# Load dữ liệu từ Open-Meteo và validate
# ================================================================

import pandas as pd
import requests
import warnings
from typing import Dict, List, Optional
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CITIES, RAW_DATA_FILE, PROCESSED_DATA_FILE,
    OPEN_METEO_URL, START_DATE, END_DATE, 
    WEATHER_VARIABLES_DAILY, WEATHER_VARIABLES_HOURLY
)

warnings.filterwarnings("ignore")


def fetch_weather_data_from_api(
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    cities: Dict = CITIES
) -> pd.DataFrame:
    """
    Lấy dữ liệu thời tiết từ Open-Meteo API cho tất cả cities.
    """
    print(f"   Đang lấy dữ liệu từ Open-Meteo API...")
    print(f"   Thời gian: {start_date} đến {end_date}")
    print(f"   Số thành phố: {len(cities)}")
    
    all_data = []
    
    for city_name, city_info in cities.items():
        print(f"   {city_name}... ", end="")
        
        params = {
            "latitude": city_info["lat"],
            "longitude": city_info["lon"],
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(WEATHER_VARIABLES_DAILY),
            "hourly": ",".join(WEATHER_VARIABLES_HOURLY),
            "timezone": "auto"
        }
        
        try:
            response = requests.get(OPEN_METEO_URL, params=params, timeout=30)
            
            # Thêm dòng này để in ra chi tiết nếu API báo lỗi thay vì chỉ hiện 400
            if response.status_code != 200:
                print(f"\n   Lỗi API chi tiết: {response.text}")
                
            response.raise_for_status()
            data = response.json()
            
            # 1. Parse dữ liệu Daily
            df_daily = pd.DataFrame({
                "Date": pd.to_datetime(data["daily"]["time"]),
                "TempMax": data["daily"]["temperature_2m_max"],
                "TempMean": data["daily"]["temperature_2m_mean"],
                "TempMin": data["daily"]["temperature_2m_min"],
                "Precipitation": data["daily"]["precipitation_sum"],
                "WindSpeed": data["daily"]["wind_speed_10m_max"],
                "City": city_name
            })
            
            # 2. Parse dữ liệu Hourly (Độ ẩm) và tính trung bình theo ngày
            df_hourly = pd.DataFrame({
                "Time": pd.to_datetime(data["hourly"]["time"]),
                "Humidity": data["hourly"]["relative_humidity_2m"]
            })
            # Tách lấy phần ngày từ Time
            df_hourly["Date"] = pd.to_datetime(df_hourly["Time"].dt.date)
            # Tính trung bình độ ẩm cho từng ngày
            df_humidity_daily = df_hourly.groupby("Date")["Humidity"].mean().reset_index()
            
            # 3. Merge dữ liệu Daily và Hourly lại với nhau
            df_city = pd.merge(df_daily, df_humidity_daily, on="Date", how="inner")
            
            all_data.append(df_city)
            print(f"{len(df_city)} records")
            
        except Exception as e:
            print(f"Lỗi: {e}")
            continue
    
    if not all_data:
        raise ValueError("Không lấy được dữ liệu từ API!")
    
    # Gộp tất cả data
    df_combined = pd.concat(all_data, ignore_index=True)
    
    print(f"\nTổng cộng: {len(df_combined)} records từ {len(cities)} cities")
    
    return df_combined


def load_raw_data(
    filepath: Optional[str] = None,
    fetch_new: bool = False
) -> pd.DataFrame:
    """
    Load dữ liệu raw từ file CSV hoặc fetch mới từ API.
    
    Parameters:
    -----------
    filepath : Optional[str]
        Đường dẫn file CSV (mặc định: RAW_DATA_FILE)
    fetch_new : bool
        True = fetch mới từ API, False = load từ file
    
    Returns:
    --------
    pd.DataFrame
        DataFrame chứa dữ liệu raw
    """
    if filepath is None:
        filepath = RAW_DATA_FILE
    
    if fetch_new or not os.path.exists(filepath):
        print("Fetching dữ liệu mới từ API...")
        df = fetch_weather_data_from_api()
        
        # Save to file
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Đã lưu dữ liệu raw: {filepath}")
    else:
        print(f"Loading dữ liệu từ file: {filepath}")
        df = pd.read_csv(filepath, parse_dates=["Date"])
        print(f"Loaded {len(df)} records")
    
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate và clean dữ liệu.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần validate
    
    Returns:
    --------
    pd.DataFrame
        DataFrame đã được validate và clean
    """
    print("\nValidating dữ liệu...")
    
    initial_rows = len(df)
    
    # Check missing values
    missing = df.isnull().sum()
    if missing.any():
        print("Missing values:")
        print(missing[missing > 0])
    
    # Drop missing values
    df = df.dropna()
    
    # Check duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Tìm thấy {duplicates} duplicates - đã xóa")
        df = df.drop_duplicates()
    
    # Validate ranges
    validation_checks = [
        ("Humidity", 0, 100),
        ("Precipitation", 0, 1000),
        ("WindSpeed", 0, 300),
        ("TempMax", -50, 60),
        ("TempMean", -50, 60),
        ("TempMin", -50, 60)
    ]
    
    for col, min_val, max_val in validation_checks:
        invalid = ((df[col] < min_val) | (df[col] > max_val)).sum()
        if invalid > 0:
            print(f"{col}: {invalid} giá trị ngoài phạm vi [{min_val}, {max_val}]")
            df = df[(df[col] >= min_val) & (df[col] <= max_val)]
    
    final_rows = len(df)
    removed = initial_rows - final_rows
    
    if removed > 0:
        print(f"\nĐã xóa {removed} rows ({removed/initial_rows*100:.2f}%)")
    
    print(f"Validation complete: {final_rows} rows còn lại")
    
    return df


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Tạo summary của dữ liệu.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần summarize
    
    Returns:
    --------
    Dict
        Dictionary chứa summary statistics
    """
    summary = {
        "total_records": len(df),
        "cities": df["City"].unique().tolist(),
        "num_cities": df["City"].nunique(),
        "date_range": {
            "start": df["Date"].min().strftime("%Y-%m-%d"),
            "end": df["Date"].max().strftime("%Y-%m-%d"),
            "days": (df["Date"].max() - df["Date"].min()).days
        },
        "temp_stats": {
            "max": {
                "min": df["TempMax"].min(),
                "max": df["TempMax"].max(),
                "mean": df["TempMax"].mean()
            },
            "mean": {
                "min": df["TempMean"].min(),
                "max": df["TempMean"].max(),
                "mean": df["TempMean"].mean()
            },
            "min": {
                "min": df["TempMin"].min(),
                "max": df["TempMin"].max(),
                "mean": df["TempMin"].mean()
            }
        },
        "missing_values": df.isnull().sum().to_dict()
    }
    
    return summary


def print_data_info(df: pd.DataFrame) -> None:
    """
    In thông tin chi tiết về dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần in thông tin
    """
    print("\n" + "="*70)
    print("DATA INFORMATION".center(70))
    print("="*70 + "\n")
    
    print(f"Tổng số records: {len(df):,}")
    print(f"Số cities: {df['City'].nunique()}")
    print(f"Khoảng thời gian: {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"Số ngày: {(df['Date'].max() - df['Date'].min()).days:,}")
    
    print("\nNHIỆT ĐỘ:")
    for target in ["TempMax", "TempMean", "TempMin"]:
        print(f"   {target:10s}: {df[target].min():6.1f}°C → {df[target].max():6.1f}°C (TB: {df[target].mean():6.1f}°C)")
    
    print("\nRECORDS PER CITY:")
    for city, count in df["City"].value_counts().items():
        print(f"   {city:12s}: {count:,} records")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Test module
    print("Testing load_data module...")
    
    # Load data
    df = load_raw_data(fetch_new=False)
    
    # Validate
    df = validate_data(df)
    
    # Print info
    print_data_info(df)
    
    # Summary
    summary = get_data_summary(df)
    print(f"\nSummary: {summary['num_cities']} cities, {summary['total_records']} records")
