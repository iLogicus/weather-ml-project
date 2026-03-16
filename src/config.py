# ================================================================
# CONFIGURATION FILE
# Tất cả constants, parameters và configs cho project
# ================================================================

import os

# ================================================================
# PATHS
# ================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models", "saved_models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# Data files
RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, "raw_weather.csv")
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "processed_weather.csv")

# ================================================================
# CITIES & CLIMATE DATA
# ================================================================
CITIES = {
    "Amsterdam": {
        "lat": 52.3676,
        "lon": 4.9041,
        "khi_hau": "Oceanic (ôn đới hải dương)",
        "mau": "#ff7b72"
    },
    "Berlin": {
        "lat": 52.5200,
        "lon": 13.4050,
        "khi_hau": "Continental (lục địa)",
        "mau": "#58a6ff"
    },
    "Athens": {
        "lat": 37.9838,
        "lon": 23.7275,
        "khi_hau": "Mediterranean (Địa Trung Hải)",
        "mau": "#ffa657"
    },
    "Stockholm": {
        "lat": 59.3293,
        "lon": 18.0686,
        "khi_hau": "Nordic (Bắc Âu)",
        "mau": "#d2a8ff"
    },
    "Zurich": {
        "lat": 47.3769,
        "lon": 8.5417,
        "khi_hau": "Alpine (núi cao)",
        "mau": "#3fb950"
    }
}

# ================================================================
# FEATURES & TARGETS
# ================================================================
FEATURE_COLUMNS = [
    "Humidity",
    "Precipitation", 
    "WindSpeed",
    "Month",
    "City_encoded",
    "DaylightHours",
    "TempLag1",
    "TempLag3"
]

TARGET_COLUMNS = ["TempMax", "TempMean", "TempMin"]

TARGET_NAMES = {
    "TempMax": "Nhiệt Độ Tối Đa",
    "TempMean": "Nhiệt Độ Trung Bình", 
    "TempMin": "Nhiệt Độ Tối Thiểu"
}

# ================================================================
# MODEL CONFIGURATIONS
# ================================================================
MODEL_CONFIGS = {
    "Ridge": {
        "params": {"alpha": 1.0},
        "display_name": "Ridge Regression"
    },
    "KNN": {
        "params": {"n_neighbors": 5, "weights": "distance"},
        "display_name": "K-Nearest Neighbors"
    },
    "Random Forest": {
        "params": {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 5,
            "random_state": 42
        },
        "display_name": "Random Forest"
    },
    "XGBoost": {
        "params": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42
        },
        "display_name": "XGBoost"
    },
    "SVR": {
        "params": {"kernel": "rbf", "C": 10, "epsilon": 0.1},
        "display_name": "Support Vector Regression"
    }
}

# Model colors cho visualization
MODEL_COLORS = {
    "Ridge": "#ff7b72",
    "KNN": "#58a6ff",
    "Random Forest": "#3fb950",
    "XGBoost": "#d2a8ff",
    "SVR": "#ffa657"
}

# ================================================================
# TRAINING PARAMETERS
# ================================================================
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ================================================================
# OPEN-METEO API CONFIGURATION
# ================================================================
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"

WEATHER_VARIABLES_DAILY = [
    "temperature_2m_max",
    "temperature_2m_mean", 
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max"
]

WEATHER_VARIABLES_HOURLY = [
    "relative_humidity_2m"
]

# ================================================================
# UI THEME COLORS (cho Streamlit)
# ================================================================
THEME_COLORS = {
    "bg": "#0d1117",
    "surface": "#161b22",
    "border": "#30363d",
    "xanh": "#58a6ff",
    "xanhla": "#3fb950",
    "do": "#ff7b72",
    "tim": "#d2a8ff",
    "cam": "#ffa657",
    "vang": "#e3b341",
    "text": "#e6edf3",
    "mo": "#8b949e"
}

# ================================================================
# VALIDATION RANGES
# ================================================================
VALIDATION_RANGES = {
    "Humidity": (0, 100),
    "Precipitation": (0, 500),
    "WindSpeed": (0, 200),
    "Month": (1, 12),
    "TempLag1": (-40, 50),
    "TempLag3": (-40, 50)
}