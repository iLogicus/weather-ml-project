# ================================================================
# PREDICTION FUNCTIONS
# Load trained models và thực hiện predictions
# ================================================================

import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CITIES, FEATURE_COLUMNS, TARGET_COLUMNS,
    MODEL_CONFIGS, MODELS_DIR
)
from utils.helpers import load_pickle, calculate_daylight_hours


def load_all_models(models_dir: str = MODELS_DIR) -> Dict:
    """
    Load tất cả trained models từ pickle files.
    
    Parameters:
    -----------
    models_dir : str
        Thư mục chứa model files
    
    Returns:
    --------
    Dict
        Dictionary chứa tất cả models: {target: {model_name: model}}
    """
    print(f"Loading models từ: {models_dir}")
    
    models = {}
    
    for target in TARGET_COLUMNS:
        models[target] = {}
        for model_name in MODEL_CONFIGS.keys():
            filepath = os.path.join(models_dir, f"{target}_{model_name}.pkl")
            try:
                model = load_pickle(filepath)
                models[target][model_name] = model
            except FileNotFoundError:
                print(f"Không tìm thấy: {filepath}")
    
    print(f"Loaded {len(TARGET_COLUMNS)} × {len(MODEL_CONFIGS)} = {len(TARGET_COLUMNS) * len(MODEL_CONFIGS)} models")
    
    return models


def load_scalers(models_dir: str = MODELS_DIR) -> Dict:
    """
    Load tất cả scalers.
    
    Parameters:
    -----------
    models_dir : str
        Thư mục chứa scaler files
    
    Returns:
    --------
    Dict
        Dictionary chứa scalers: {target: scaler}
    """
    scalers = {}
    
    for target in TARGET_COLUMNS:
        filepath = os.path.join(models_dir, f"scaler_{target}.pkl")
        try:
            scaler = load_pickle(filepath)
            scalers[target] = scaler
        except FileNotFoundError:
            print(f"Không tìm thấy scaler: {filepath}")
    
    return scalers


def load_lag_metadata(models_dir: str = MODELS_DIR) -> pd.DataFrame:
    """
    Load lag statistics metadata.
    
    Parameters:
    -----------
    models_dir : str
        Thư mục chứa metadata file
    
    Returns:
    --------
    pd.DataFrame
        DataFrame chứa lag statistics
    """
    filepath = os.path.join(models_dir, "lag_metadata.pkl")
    try:
        lag_metadata = load_pickle(filepath)
        return lag_metadata
    except FileNotFoundError:
        print(f"Không tìm thấy lag metadata: {filepath}")
        return None


def load_evaluation_metrics(models_dir: str = MODELS_DIR) -> Dict:
    """
    Load evaluation metrics.
    
    Parameters:
    -----------
    models_dir : str
        Thư mục chứa metrics file
    
    Returns:
    --------
    Dict
        Dictionary chứa metrics
    """
    filepath = os.path.join(models_dir, "evaluation_metrics.pkl")
    try:
        metrics = load_pickle(filepath)
        return metrics
    except FileNotFoundError:
        print(f"Không tìm thấy evaluation metrics: {filepath}")
        return None


def prepare_input_features(
    city: str,
    month: int,
    humidity: float,
    precipitation: float,
    wind_speed: float,
    lag_metadata: pd.DataFrame = None,
    cities: Dict = CITIES
) -> np.ndarray:
    """
    Chuẩn bị input features từ user input.
    
    Parameters:
    -----------
    city : str
        Tên thành phố
    month : int
        Tháng (1-12)
    humidity : float
        Độ ẩm (%)
    precipitation : float
        Lượng mưa (mm)
    wind_speed : float
        Tốc độ gió (km/h)
    lag_metadata : pd.DataFrame
        Lag statistics (optional)
    cities : Dict
        Dictionary chứa thông tin cities
    
    Returns:
    --------
    np.ndarray
        Feature vector (1 × 8)
    """
    # City encoding (alphabetical order)
    city_encoded = sorted(cities.keys()).index(city)
    
    # Daylight hours (ngày 15 của tháng)
    day_of_year = pd.Timestamp(2024, month, 15).dayofyear
    latitude = cities[city]["lat"]
    daylight_hours = calculate_daylight_hours(latitude, day_of_year)
    
    # Lag features từ metadata
    if lag_metadata is not None:
        try:
            temp_lag1 = lag_metadata.loc[(city, month), "TempLag1"]
            temp_lag3 = lag_metadata.loc[(city, month), "TempLag3"]
        except KeyError:
            # Fallback: dùng mean của city
            city_data = lag_metadata.loc[city]
            temp_lag1 = city_data["TempLag1"].mean()
            temp_lag3 = city_data["TempLag3"].mean()
    else:
        # Default values
        temp_lag1 = 10.0
        temp_lag3 = 10.0
    
    # Tạo feature vector theo thứ tự FEATURE_COLUMNS
    # ["Humidity", "Precipitation", "WindSpeed", "Month", "City_encoded",
    #  "DaylightHours", "TempLag1", "TempLag3"]
    features = np.array([[
        humidity,
        precipitation,
        wind_speed,
        month,
        city_encoded,
        daylight_hours,
        temp_lag1,
        temp_lag3
    ]], dtype=float)
    
    return features


def predict_temperature(
    city: str,
    month: int,
    humidity: float,
    precipitation: float,
    wind_speed: float,
    model_name: str,
    models: Dict,
    scalers: Dict,
    lag_metadata: pd.DataFrame = None
) -> Dict[str, float]:
    """
    Dự đoán nhiệt độ cho tất cả targets.
    
    Parameters:
    -----------
    city : str
        Tên thành phố
    month : int
        Tháng (1-12)
    humidity : float
        Độ ẩm (%)
    precipitation : float
        Lượng mưa (mm)
    wind_speed : float
        Tốc độ gió (km/h)
    model_name : str
        Tên model để dùng
    models : Dict
        Dictionary chứa trained models
    scalers : Dict
        Dictionary chứa scalers
    lag_metadata : pd.DataFrame
        Lag statistics
    
    Returns:
    --------
    Dict[str, float]
        {"TempMax": value, "TempMean": value, "TempMin": value}
    """
    # Prepare features
    features = prepare_input_features(
        city, month, humidity, precipitation, wind_speed, lag_metadata
    )
    
    predictions = {}
    
    for target in TARGET_COLUMNS:
        # Scale features
        scaler = scalers[target]
        features_scaled = scaler.transform(features)
        
        # Predict
        model = models[target][model_name]
        pred = model.predict(features_scaled)[0]
        
        predictions[target] = round(pred, 1)
    
    return predictions


def get_best_model_for_target(
    target: str,
    metrics: Dict
) -> Tuple[str, float]:
    """
    Lấy model tốt nhất cho một target dựa trên R².
    
    Parameters:
    -----------
    target : str
        Target column
    metrics : Dict
        Evaluation metrics
    
    Returns:
    --------
    Tuple[str, float]
        (best_model_name, best_r2_score)
    """
    target_metrics = metrics[target]
    
    best_model = max(
        target_metrics.keys(),
        key=lambda m: target_metrics[m]["R2"]
    )
    
    best_r2 = target_metrics[best_model]["R2"]
    
    return best_model, best_r2


if __name__ == "__main__":
    # Test prediction
    print("Testing prediction module...")
    
    # Load models
    models = load_all_models()
    scalers = load_scalers()
    lag_metadata = load_lag_metadata()
    metrics = load_evaluation_metrics()
    
    # Test prediction
    test_input = {
        "city": "Amsterdam",
        "month": 6,
        "humidity": 70,
        "precipitation": 2.0,
        "wind_speed": 20.0,
        "model_name": "XGBoost"
    }
    
    print(f"\nTest prediction:")
    print(f"   Input: {test_input}")
    
    predictions = predict_temperature(
        **test_input,
        models=models,
        scalers=scalers,
        lag_metadata=lag_metadata
    )
    
    print(f"\nPredictions:")
    for target, value in predictions.items():
        print(f"   {target}: {value}°C")
    
    # Best models
    print(f"\nBest models by target:")
    for target in TARGET_COLUMNS:
        best_model, best_r2 = get_best_model_for_target(target, metrics)
        print(f"   {target}: {MODEL_CONFIGS[best_model]['display_name']} (R²={best_r2:.4f})")
