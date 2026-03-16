# ================================================================
# HELPER FUNCTIONS
# Các hàm tiện ích dùng chung cho project
# ================================================================

import numpy as np
import math
import os
import pickle
from typing import Any, Dict, List, Tuple


def calculate_daylight_hours(latitude: float, day_of_year: int) -> float:
    """
    Tính số giờ chiếu sáng dựa trên vĩ độ và ngày trong năm.
    
    Parameters:
    -----------
    latitude : float
        Vĩ độ của địa điểm (độ)
    day_of_year : int
        Ngày thứ mấy trong năm (1-365)
    
    Returns:
    --------
    float
        Số giờ chiếu sáng
    """
    lat_rad = math.radians(latitude)
    
    # Góc nghiêng của Trái Đất
    tilt = 23.44
    tilt_rad = math.radians(tilt)
    
    # Góc declination của mặt trời
    declination = tilt_rad * math.sin(math.radians(360 / 365 * (day_of_year - 81)))
    
    # Góc giờ mặt mọc/lặn
    try:
        cos_hour_angle = -math.tan(lat_rad) * math.tan(declination)
        # Giới hạn trong [-1, 1] để tránh lỗi acos
        cos_hour_angle = max(-1, min(1, cos_hour_angle))
        hour_angle = math.acos(cos_hour_angle)
        daylight = (2 * hour_angle * 24) / (2 * math.pi)
        return round(daylight, 2)
    except:
        # Trường hợp cực (ngày cực hoặc đêm cực)
        if latitude > 66.5 and day_of_year > 80 and day_of_year < 264:
            return 24.0  # Ngày cực
        elif latitude > 66.5:
            return 0.0   # Đêm cực
        else:
            return 12.0  # Mặc định


def save_pickle(obj: Any, filepath: str) -> None:
    """
    Lưu object vào file pickle.
    
    Parameters:
    -----------
    obj : Any
        Object cần lưu
    filepath : str
        Đường dẫn file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Đã lưu: {filepath}")


def load_pickle(filepath: str) -> Any:
    """
    Load object từ file pickle.
    
    Parameters:
    -----------
    filepath : str
        Đường dẫn file
    
    Returns:
    --------
    Any
        Object đã load
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")
    
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    return obj


def create_directory_if_not_exists(directory: str) -> None:
    """
    Tạo thư mục nếu chưa tồn tại.
    
    Parameters:
    -----------
    directory : str
        Đường dẫn thư mục
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Đã tạo thư mục: {directory}")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Tính các metrics đánh giá model.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Giá trị thực tế
    y_pred : np.ndarray
        Giá trị dự đoán
    
    Returns:
    --------
    Dict[str, float]
        Dictionary chứa MAE, MSE, RMSE, R2
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    }


def validate_input_ranges(data: Dict[str, float], ranges: Dict[str, Tuple[float, float]]) -> Tuple[bool, List[str]]:
    """
    Validate input data trong phạm vi cho phép.
    
    Parameters:
    -----------
    data : Dict[str, float]
        Dictionary chứa dữ liệu cần validate
    ranges : Dict[str, Tuple[float, float]]
        Dictionary chứa phạm vi (min, max) cho mỗi feature
    
    Returns:
    --------
    Tuple[bool, List[str]]
        (is_valid, error_messages)
    """
    errors = []
    
    for key, value in data.items():
        if key in ranges:
            min_val, max_val = ranges[key]
            if not (min_val <= value <= max_val):
                errors.append(f"{key} phải nằm trong khoảng [{min_val}, {max_val}], nhận được {value}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def format_temperature(temp: float, unit: str = "C") -> str:
    """
    Format nhiệt độ với đơn vị.
    
    Parameters:
    -----------
    temp : float
        Giá trị nhiệt độ
    unit : str
        Đơn vị ('C' hoặc 'F')
    
    Returns:
    --------
    str
        Chuỗi nhiệt độ đã format
    """
    if unit == "C":
        return f"{temp:.1f}°C"
    elif unit == "F":
        fahrenheit = (temp * 9/5) + 32
        return f"{fahrenheit:.1f}°F"
    else:
        return f"{temp:.1f}°{unit}"


def get_season_from_month(month: int) -> str:
    """
    Xác định mùa từ tháng (cho bán cầu Bắc).
    
    Parameters:
    -----------
    month : int
        Tháng (1-12)
    
    Returns:
    --------
    str
        Tên mùa
    """
    if month in [12, 1, 2]:
        return "Mùa đông"
    elif month in [3, 4, 5]:
        return "Mùa xuân"
    elif month in [6, 7, 8]:
        return "Mùa hè"
    else:
        return "Mùa thu"


def print_section_header(title: str, width: int = 70) -> None:
    """
    In tiêu đề section đẹp.
    
    Parameters:
    -----------
    title : str
        Tiêu đề
    width : int
        Độ rộng
    """
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width + "\n")
