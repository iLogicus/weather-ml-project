import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý các vấn đề cơ bản về dữ liệu thô: trùng lặp và khuyết thiếu.
    """
    df_cleaned = df.copy()
    
    # 1. Xử lý trùng lặp (Duplicate Records)
    initial_count = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates()
    final_count = len(df_cleaned)
    if initial_count > final_count:
        logging.info(f"Đã loại bỏ {initial_count - final_count} bản ghi trùng lặp.")
    
    # 2. Xử lý giá trị khuyết thiếu (Missing Values) bằng Forward Fill
    # Chỉ áp dụng cho các cột số (khí tượng)
    null_count = df_cleaned.isnull().sum().sum()
    if null_count > 0:
        df_cleaned = df_cleaned.ffill()
        logging.info(f"Đã xử lý {null_count} giá trị khuyết bằng phương pháp Forward Fill.")
        
    return df_cleaned

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý ngoại lai dựa trên các ngưỡng vật lý logic tại Châu Âu.
    """
    df_out = df.copy()
    
    # Định nghĩa ngưỡng vật lý (có thể đưa vào config.py sau này)
    PHYSICAL_THRESHOLDS = {
        'temp_max': (-60, 60),
        'temp_mean': (-60, 60),
        'temp_min': (-60, 60),
        'relative_humidity_2m': (0, 100),
        'wind_speed_10m': (0, 250) # Tốc độ gió bão cực đại
    }
    
    for col, (min_val, max_val) in PHYSICAL_THRESHOLDS.items():
        if col in df_out.columns:
            # Lọc bỏ các dòng vi phạm ngưỡng vật lý
            before_count = len(df_out)
            df_out = df_out[(df_out[col] >= min_val) & (df_out[col] <= max_val)]
            diff = before_count - len(df_out)
            if diff > 0:
                logging.warning(f"⚠️ Đã loại bỏ {diff} bản ghi ngoại lai tại cột '{col}'.")
                
    return df_out

def validate_data_status(df: pd.DataFrame) -> bool:
    """
    Hàm xác thực trạng thái 'Sạch' của dữ liệu (Read-only).
    Trả về True nếu dữ liệu đạt chuẩn để Feature Engineering.
    """
    has_nulls = df.isnull().any().any()
    has_duplicates = df.duplicated().any()
    
    if has_nulls:
        logging.error("Dữ liệu vẫn còn giá trị rỗng (Null).")
    if has_duplicates:
        logging.error("Dữ liệu vẫn còn bản ghi trùng lặp.")
        
    return not (has_nulls or has_duplicates)