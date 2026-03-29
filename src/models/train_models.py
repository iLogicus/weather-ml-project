# ================================================================
# MODEL TRAINING PIPELINE
# Train tất cả models và lưu vào pickle files
# ================================================================

import pandas as pd
import numpy as np
import sys
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    FEATURE_COLUMNS, TARGET_COLUMNS, MODEL_CONFIGS,
    TEST_SIZE, RANDOM_STATE, MODELS_DIR
)
from utils.helpers import save_pickle, calculate_metrics, print_section_header

warnings.filterwarnings("ignore")


def get_model_instance(model_name: str):
    """
    Tạo instance của model dựa trên tên.
    
    Parameters:
    -----------
    model_name : str
        Tên model
    
    Returns:
    --------
    model instance
    """
    config = MODEL_CONFIGS[model_name]
    params = config["params"]
    
    if model_name == "Ridge":
        return Ridge(**params)
    elif model_name == "KNN":
        return KNeighborsRegressor(**params)
    elif model_name == "Random Forest":
        return RandomForestRegressor(**params)
    elif model_name == "XGBoost":
        return XGBRegressor(**params)
    elif model_name == "SVR":
        return SVR(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def prepare_data(df: pd.DataFrame, feature_cols: list, target_col: str):
    """
    Chuẩn bị data cho training.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame đã processed
    feature_cols : list
        Danh sách tên features
    target_col : str
        Tên target column
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, scaler, test_cities)
    """
    # Extract features và target
    X = df[feature_cols].values
    y = df[target_col].values
    cities = df["City"].values
    
    # Split train/test
    X_train, X_test, y_train, y_test, cities_train, cities_test = train_test_split(
        X, y, cities, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, cities_test


def train_single_model(
    model_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cities_test: np.ndarray
) -> dict:
    """
    Train một model duy nhất.
    
    Parameters:
    -----------
    model_name : str
        Tên model
    X_train, X_test : np.ndarray
        Training và test features
    y_train, y_test : np.ndarray
        Training và test targets
    cities_test : np.ndarray
        Cities tương ứng với test set
    
    Returns:
    --------
    dict
        Dictionary chứa model và results
    """
    print(f"   Training {model_name}... ", end="")
    
    # Get model instance
    model = get_model_instance(model_name)
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    print(f"R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}")
    
    return {
        "model": model,
        "metrics": metrics,
        "y_true": y_test,
        "y_pred": y_pred,
        "cities_test": cities_test
    }


def train_all_models_for_target(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list = FEATURE_COLUMNS
) -> dict:
    """
    Train tất cả models cho một target.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame đã processed
    target_col : str
        Tên target column
    feature_cols : list
        Danh sách tên features
    
    Returns:
    --------
    dict
        Dictionary chứa results của tất cả models
    """
    print(f"\nTARGET: {target_col}")
    print("="*70)
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler, cities_test = prepare_data(
        df, feature_cols, target_col
    )
    
    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    
    # Train all models
    results = {}
    for model_name in MODEL_CONFIGS.keys():
        result = train_single_model(
            model_name, X_train, X_test, y_train, y_test, cities_test
        )
        results[model_name] = result
    
    # Add scaler to results
    results["scaler"] = scaler
    
    return results


def train_all_models(df: pd.DataFrame) -> dict:
    """
    Train tất cả models cho tất cả targets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame đã processed
    
    Returns:
    --------
    dict
        Dictionary chứa results của tất cả targets × models
    """
    print_section_header("TRAINING TẤT CẢ MODELS")
    
    all_results = {}
    
    for target in TARGET_COLUMNS:
        results = train_all_models_for_target(df, target)
        all_results[target] = results
    
    print_section_header("HOÀN THÀNH TRAINING TẤT CẢ MODELS")
    
    return all_results


def save_all_models(all_results: dict, models_dir: str = MODELS_DIR) -> None:
    """
    Lưu tất cả models và metadata vào pickle files.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary chứa results từ train_all_models()
    models_dir : str
        Thư mục lưu models
    """
    print_section_header("LƯU TẤT CẢ MODELS")
    
    os.makedirs(models_dir, exist_ok=True)
    
    # Save models
    for target in TARGET_COLUMNS:
        for model_name in MODEL_CONFIGS.keys():
            model = all_results[target][model_name]["model"]
            filepath = os.path.join(models_dir, f"{target}_{model_name}.pkl")
            save_pickle(model, filepath)
    
    # Save scalers (one per target)
    for target in TARGET_COLUMNS:
        scaler = all_results[target]["scaler"]
        filepath = os.path.join(models_dir, f"scaler_{target}.pkl")
        save_pickle(scaler, filepath)
    
    # Save evaluation metrics
    metrics_dict = {}
    for target in TARGET_COLUMNS:
        metrics_dict[target] = {}
        for model_name in MODEL_CONFIGS.keys():
            metrics_dict[target][model_name] = all_results[target][model_name]["metrics"]
    
    metrics_filepath = os.path.join(models_dir, "evaluation_metrics.pkl")
    save_pickle(metrics_dict, metrics_filepath)
    
    print(f"\nĐã lưu tất cả models vào: {models_dir}")
    print(f"   {len(TARGET_COLUMNS)} targets × {len(MODEL_CONFIGS)} models = {len(TARGET_COLUMNS) * len(MODEL_CONFIGS)} model files")
    print(f"   {len(TARGET_COLUMNS)} scaler files")
    print(f"   1 evaluation metrics file")


def print_evaluation_summary(all_results: dict) -> None:
    """
    In summary của evaluation results.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary chứa results
    """
    print_section_header("EVALUATION SUMMARY")
    
    for target in TARGET_COLUMNS:
        print(f"\n{target}")
        print("-" * 70)
        print(f"{'Model':<25} {'R²':>10} {'MAE':>10} {'RMSE':>10}")
        print("-" * 70)
        
        # Sort by R2
        results = all_results[target]
        sorted_models = sorted(
            MODEL_CONFIGS.keys(),
            key=lambda m: results[m]["metrics"]["R2"],
            reverse=True
        )
        
        for model_name in sorted_models:
            metrics = results[model_name]["metrics"]
            display_name = MODEL_CONFIGS[model_name]["display_name"]
            print(f"{display_name:<25} {metrics['R2']:>10.4f} {metrics['MAE']:>10.4f} {metrics['RMSE']:>10.4f}")
        
        # Best model
        best_model = sorted_models[0]
        best_r2 = results[best_model]["metrics"]["R2"]
        print(f"\n   Best: {MODEL_CONFIGS[best_model]['display_name']} (R²={best_r2:.4f})")


if __name__ == "__main__":
    # Main training pipeline
    print("STARTING TRAINING PIPELINE...")
    
    from features.feature_engineering import load_processed_data, calculate_lag_statistics
    
    # Load processed data
    df = load_processed_data()
    
    # Calculate and save lag statistics
    lag_stats = calculate_lag_statistics(df)
    lag_stats_path = os.path.join(MODELS_DIR, "lag_metadata.pkl")
    save_pickle(lag_stats, lag_stats_path)
    
    # Train all models
    all_results = train_all_models(df)
    
    # Print summary
    print_evaluation_summary(all_results)
    
    # Save models
    save_all_models(all_results)
    
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE!".center(70))
    print("="*70)
