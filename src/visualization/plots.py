# ================================================================
# VISUALIZATION FUNCTIONS
# Các hàm vẽ biểu đồ cho Streamlit app
# ================================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CITIES, MODEL_COLORS, TARGET_NAMES


def setup_dark_theme():
    """Setup matplotlib dark theme."""
    plt.style.use('dark_background')
    
    # Custom colors
    plt.rcParams.update({
        'figure.facecolor': '#0d1117',
        'axes.facecolor': '#161b22',
        'axes.edgecolor': '#30363d',
        'axes.labelcolor': '#e6edf3',
        'xtick.color': '#e6edf3',
        'ytick.color': '#e6edf3',
        'text.color': '#e6edf3',
        'grid.color': '#30363d',
        'grid.alpha': 0.3,
        'legend.facecolor': '#161b22',
        'legend.edgecolor': '#30363d'
    })


def plot_temperature_by_city(df: pd.DataFrame, figsize=(12, 4)):
    """
    Vẽ biểu đồ nhiệt độ theo thành phố.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu
    figsize : tuple
        Kích thước figure
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    setup_dark_theme()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    cities = sorted(df["City"].unique())
    x_pos = np.arange(len(cities))
    width = 0.25
    
    temp_max_means = [df[df["City"]==city]["TempMax"].mean() for city in cities]
    temp_mean_means = [df[df["City"]==city]["TempMean"].mean() for city in cities]
    temp_min_means = [df[df["City"]==city]["TempMin"].mean() for city in cities]
    
    ax.bar(x_pos - width, temp_max_means, width, label='Tối đa', 
           color='#ff7b72', alpha=0.85)
    ax.bar(x_pos, temp_mean_means, width, label='Trung bình',
           color='#58a6ff', alpha=0.85)
    ax.bar(x_pos + width, temp_min_means, width, label='Tối thiểu',
           color='#d2a8ff', alpha=0.85)
    
    ax.set_xlabel('Thành phố')
    ax.set_ylabel('Nhiệt độ (°C)')
    ax.set_title('Nhiệt Độ Trung Bình Theo Thành Phố', fontsize=13, pad=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cities)
    ax.legend(framealpha=0.3)
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    return fig


def plot_temperature_distribution(df: pd.DataFrame, target: str, figsize=(10, 4)):
    """
    Vẽ phân phối nhiệt độ cho một target.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu
    target : str
        Target column
    figsize : tuple
        Kích thước figure
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    setup_dark_theme()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Boxplot
    ax = axes[0]
    cities = sorted(df["City"].unique())
    data_by_city = [df[df["City"]==city][target].values for city in cities]
    
    bp = ax.boxplot(data_by_city, labels=cities, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#58a6ff')
        patch.set_alpha(0.6)
    
    ax.set_ylabel('°C')
    ax.set_title(f'Phân Phối {TARGET_NAMES[target]}', fontsize=11)
    ax.grid(True, axis='y')
    
    # Violin plot
    ax = axes[1]
    for i, city in enumerate(cities):
        city_data = df[df["City"]==city][target].values
        parts = ax.violinplot([city_data], positions=[i], widths=0.6,
                               showmeans=True, showmedians=True)
        
        for pc in parts['bodies']:
            pc.set_facecolor(CITIES[city]["mau"])
            pc.set_alpha(0.5)
    
    ax.set_xticks(range(len(cities)))
    ax.set_xticklabels(cities)
    ax.set_ylabel('°C')
    ax.set_title(f'Violin Plot - {TARGET_NAMES[target]}', fontsize=11)
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, figsize=(10, 8)):
    """
    Vẽ correlation heatmap.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu
    figsize : tuple
        Kích thước figure
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    setup_dark_theme()
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude Date-related columns
    exclude_cols = ['Year', 'Day', 'DayOfYear', 'Quarter']
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    corr = df[numerical_cols].corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Custom colormap
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    
    sns.heatmap(corr, annot=True, fmt='.2f', cmap=cmap, center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                ax=ax)
    
    ax.set_title('Correlation Matrix', fontsize=13, pad=10)
    
    plt.tight_layout()
    return fig


def plot_seasonal_patterns(df: pd.DataFrame, city: str, figsize=(10, 4)):
    """
    Vẽ seasonal patterns cho một city.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu
    city : str
        Tên thành phố
    figsize : tuple
        Kích thước figure
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    setup_dark_theme()
    
    city_data = df[df["City"] == city].groupby("Month").agg({
        "TempMax": "mean",
        "TempMean": "mean",
        "TempMin": "mean"
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(city_data["Month"], city_data["TempMax"], 
            marker='o', label='Tối đa', color='#ff7b72', linewidth=2)
    ax.plot(city_data["Month"], city_data["TempMean"], 
            marker='s', label='Trung bình', color='#58a6ff', linewidth=2)
    ax.plot(city_data["Month"], city_data["TempMin"], 
            marker='^', label='Tối thiểu', color='#d2a8ff', linewidth=2)
    
    ax.fill_between(city_data["Month"], city_data["TempMin"], 
                     city_data["TempMax"], alpha=0.15, color='#58a6ff')
    
    ax.set_xlabel('Tháng')
    ax.set_ylabel('Nhiệt độ (°C)')
    ax.set_title(f'Seasonal Patterns - {city}', fontsize=13, pad=10)
    ax.set_xticks(range(1, 13))
    ax.legend(framealpha=0.3)
    ax.grid(True)
    
    plt.tight_layout()
    return fig


def plot_model_comparison(metrics: dict, target: str, figsize=(10, 4)):
    """
    Vẽ biểu đồ so sánh các models.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary chứa evaluation metrics
    target : str
        Target column
    figsize : tuple
        Kích thước figure
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    setup_dark_theme()
    
    from config import MODEL_CONFIGS
    
    model_names = list(MODEL_CONFIGS.keys())
    display_names = [MODEL_CONFIGS[m]["display_name"] for m in model_names]
    colors = [MODEL_COLORS[m] for m in model_names]
    
    r2_scores = [metrics[target][m]["R2"] for m in model_names]
    mae_scores = [metrics[target][m]["MAE"] for m in model_names]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # R² scores
    ax = axes[0]
    bars = ax.barh(display_names, r2_scores, color=colors, alpha=0.85, height=0.6)
    
    for bar, score in zip(bars, r2_scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{score:.4f}", va='center', ha='left', fontsize=9)
    
    ax.set_xlabel('R² Score')
    ax.set_title(f'R² Scores - {TARGET_NAMES[target]}', fontsize=11)
    ax.set_xlim([0, 1.0])
    ax.grid(True, axis='x')
    
    # MAE scores
    ax = axes[1]
    bars = ax.barh(display_names, mae_scores, color=colors, alpha=0.85, height=0.6)
    
    for bar, score in zip(bars, mae_scores):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{score:.3f}°C", va='center', ha='left', fontsize=9)
    
    ax.set_xlabel('MAE (°C)')
    ax.set_title(f'MAE - {TARGET_NAMES[target]}', fontsize=11)
    ax.grid(True, axis='x')
    
    plt.tight_layout()
    return fig


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cities_test: np.ndarray,
    model_name: str,
    target: str,
    r2_score: float,
    figsize=(8, 6)
):
    """
    Vẽ scatter plot actual vs predicted.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Giá trị thực tế
    y_pred : np.ndarray
        Giá trị dự đoán
    cities_test : np.ndarray
        Cities tương ứng với test samples
    model_name : str
        Tên model
    target : str
        Target column
    r2_score : float
        R² score
    figsize : tuple
        Kích thước figure
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    setup_dark_theme()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot by city
    for city in sorted(np.unique(cities_test)):
        mask = cities_test == city
        ax.scatter(y_true[mask], y_pred[mask],
                   color=CITIES[city]["mau"], alpha=0.5, s=30, label=city)
    
    # Perfect prediction line
    lims = [min(y_true.min(), y_pred.min()) - 1,
            max(y_true.max(), y_pred.max()) + 1]
    ax.plot(lims, lims, 'w--', linewidth=1.5, alpha=0.6, label='Perfect')
    
    ax.set_xlabel('Thực tế (°C)')
    ax.set_ylabel('Dự đoán (°C)')
    ax.set_title(f'{model_name} - {TARGET_NAMES[target]}\nR² = {r2_score:.4f}',
                 fontsize=12, pad=10)
    ax.legend(fontsize=8, framealpha=0.3, loc='best')
    ax.grid(True)
    
    plt.tight_layout()
    return fig


def plot_residuals_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    target: str,
    figsize=(10, 4)
):
    """
    Vẽ phân phối residuals.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Giá trị thực tế
    y_pred : np.ndarray
        Giá trị dự đoán
    model_name : str
        Tên model
    target : str
        Target column
    figsize : tuple
        Kích thước figure
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    setup_dark_theme()
    
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    ax = axes[0]
    ax.hist(residuals, bins=50, color='#58a6ff', alpha=0.7, edgecolor='white')
    ax.axvline(0, color='white', linewidth=2, linestyle='--', alpha=0.8)
    ax.set_xlabel('Residuals (°C)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Residuals Distribution - {model_name}', fontsize=11)
    ax.grid(True)
    
    # Residual plot
    ax = axes[1]
    ax.scatter(y_pred, residuals, alpha=0.5, s=20, color='#58a6ff')
    ax.axhline(0, color='white', linewidth=2, linestyle='--', alpha=0.8)
    ax.set_xlabel('Predicted (°C)')
    ax.set_ylabel('Residuals (°C)')
    ax.set_title(f'Residual Plot - {TARGET_NAMES[target]}', fontsize=11)
    ax.grid(True)
    
    plt.tight_layout()
    return fig


def plot_feature_importance(
    model,
    feature_names: list,
    model_name: str,
    target: str,
    figsize=(10, 5)
):
    """
    Vẽ feature importance (cho tree-based models).
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_
    feature_names : list
        Danh sách tên features
    model_name : str
        Tên model
    target : str
        Target column
    figsize : tuple
        Kích thước figure
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    setup_dark_theme()
    
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importances = model.feature_importances_
    
    # Sort by importance
    indices = np.argsort(importances)
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    # Color: highlight top feature
    colors = ['#21262d'] * len(sorted_features)
    colors[-1] = '#58a6ff'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.barh(sorted_features, sorted_importances, color=colors, height=0.6)
    ax.set_xlabel('Importance')
    ax.set_title(f'Feature Importance - {model_name} ({TARGET_NAMES[target]})',
                 fontsize=12, pad=10)
    ax.grid(True, axis='x')
    
    plt.tight_layout()
    return fig


def plot_prediction_comparison(
    historical_data: dict,
    predictions: dict,
    city: str,
    month: int,
    figsize=(8, 4)
):
    """
    So sánh predictions với historical averages.
    
    Parameters:
    -----------
    historical_data : dict
        {"TempMax": value, "TempMean": value, "TempMin": value}
    predictions : dict
        {"TempMax": value, "TempMean": value, "TempMin": value}
    city : str
        Tên thành phố
    month : int
        Tháng
    figsize : tuple
        Kích thước figure
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    setup_dark_theme()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    targets = ['TempMax', 'TempMean', 'TempMin']
    labels = ['Tối đa', 'Trung bình', 'Tối thiểu']
    colors = ['#ff7b72', '#58a6ff', '#d2a8ff']
    
    x = np.arange(len(targets))
    width = 0.35
    
    hist_vals = [historical_data[t] for t in targets]
    pred_vals = [predictions[t] for t in targets]
    
    # Historical bars
    ax.bar(x - width/2, hist_vals, width, label='Lịch sử TB',
           color='#30363d', alpha=0.9)
    
    # Prediction bars
    for i, (val, color) in enumerate(zip(pred_vals, colors)):
        ax.bar(x[i] + width/2, val, width, color=color, alpha=0.85)
        ax.text(x[i] + width/2, val + 0.3, f"{val:.1f}°C",
                ha='center', fontsize=9, fontweight='bold')
    
    # Historical labels
    for i, val in enumerate(hist_vals):
        ax.text(x[i] - width/2, val + 0.3, f"{val:.1f}°C",
                ha='center', fontsize=9, color='#8b949e')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('°C')
    ax.set_title(f'Dự Báo vs Lịch Sử - {city} (Tháng {month})',
                 fontsize=12, pad=10)
    ax.legend(['Lịch sử TB', 'Dự báo'], fontsize=9, framealpha=0.3)
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    return fig


def create_summary_table(metrics: dict, target: str) -> pd.DataFrame:
    """
    Tạo bảng tổng hợp metrics.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary chứa evaluation metrics
    target : str
        Target column
    
    Returns:
    --------
    pd.DataFrame
        DataFrame chứa metrics table
    """
    from config import MODEL_CONFIGS
    
    rows = []
    for model_name in MODEL_CONFIGS.keys():
        row = {
            "Model": MODEL_CONFIGS[model_name]["display_name"],
            "R²": metrics[target][model_name]["R2"],
            "MAE": metrics[target][model_name]["MAE"],
            "RMSE": metrics[target][model_name]["RMSE"]
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values("R²", ascending=False).reset_index(drop=True)
    
    return df

