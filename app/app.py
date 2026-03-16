# ================================================================
# WEATHER FORECASTING APP - STREAMLIT
# Optimized version với model persistence
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import (
    CITIES, TARGET_COLUMNS, TARGET_NAMES, MODEL_CONFIGS,
    PROCESSED_DATA_FILE
)
from models.predict import (
    load_all_models, load_scalers, load_lag_metadata,
    load_evaluation_metrics, predict_temperature,
    get_best_model_for_target
)
from visualization.plots import (
    setup_dark_theme, plot_temperature_by_city,
    plot_temperature_distribution, plot_correlation_heatmap,
    plot_seasonal_patterns, plot_model_comparison,
    plot_actual_vs_predicted, plot_feature_importance,
    plot_prediction_comparison, create_summary_table
)

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Dự Báo Thời Tiết Châu Âu",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# LOAD MODELS & DATA (with caching)
# ================================================================

@st.cache_resource
def load_models_and_metadata():
    """Load tất cả models, scalers, và metadata."""
    models = load_all_models()
    scalers = load_scalers()
    lag_metadata = load_lag_metadata()
    metrics = load_evaluation_metrics()
    return models, scalers, lag_metadata, metrics


@st.cache_data
def load_processed_df():
    """Load processed data."""
    df = pd.read_csv(PROCESSED_DATA_FILE, parse_dates=["Date"])
    return df


# Load tất cả cùng lúc
with st.spinner("Đang tải models và dữ liệu..."):
    try:
        models, scalers, lag_metadata, metrics = load_models_and_metadata()
        df = load_processed_df()
        models_loaded = True
    except Exception as e:
        st.error(f"Lỗi khi load models: {e}")
        st.info("Chạy `python src/models/train_models.py` để train models trước!")
        models_loaded = False

# ================================================================
# CUSTOM CSS (same as original)
# ================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:      #0d1117;
    --surface: #161b22;
    --border:  #30363d;
    --xanh:    #58a6ff;
    --xanhla:  #3fb950;
    --do:      #ff7b72;
    --tim:     #d2a8ff;
    --cam:     #ffa657;
    --vang:    #e3b341;
    --text:    #e6edf3;
    --mo:      #8b949e;
}
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}

.dash-header {
    background: linear-gradient(135deg, #0d1117 0%, #1a1f29 50%, #0d1117 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
}
.dash-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2.1rem; font-weight: 700;
    background: linear-gradient(90deg, #58a6ff, #79c0ff, #a5d6ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem 0;
}
.dash-header p { color: var(--mo); margin: 0; font-size: 0.92rem; }

.kpi-grid { display: grid; grid-template-columns: repeat(5,1fr); gap: .8rem; margin-bottom: 1.5rem; }
.kpi-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 1rem 1.2rem; transition: border-color .2s;
}
.kpi-card:hover { border-color: var(--xanh); }
.kpi-nhan { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 1.5px; color: var(--mo); }
.kpi-gia  { font-family: 'Space Mono', monospace; font-size: 1.6rem; font-weight: 700; }
.kpi-phu  { font-size: 0.7rem; color: var(--mo); }
.xanh { color: var(--xanh); } 
.xanhla { color: var(--xanhla); }
.do   { color: var(--do); } 
.tim  { color: var(--tim); }
.cam  { color: var(--cam); }

.tieu-de-muc {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 2px;
    color: var(--mo); border-bottom: 1px solid var(--border);
    padding-bottom: .45rem; margin: 1.4rem 0 .9rem 0;
}

.khung-dubao {
    background: linear-gradient(135deg, rgba(88,166,255,.1), rgba(210,168,255,.05));
    border: 1px solid rgba(88,166,255,.3); border-radius: 12px;
    padding: 1.5rem; text-align: center;
}
.nhiet-do-dubao { font-family: 'Space Mono', monospace; font-size: 3rem; font-weight: 700; color: var(--xanh); }
.nhan-dubao { font-size: .82rem; color: var(--mo); }
</style>
""", unsafe_allow_html=True)

# ================================================================
# SIDEBAR NAVIGATION
# ================================================================
with st.sidebar:
    st.markdown("### NAVIGATION")
    
    page = st.radio(
        "Chọn trang:",
        ["Tổng Quan", "Phân Tích Dữ Liệu", 
         "Đánh Giá Mô Hình", "Dự Báo Nhiệt Độ"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### THÔNG TIN")
    st.info(f"""
    **Dataset**: {len(df):,} records  
    **Cities**: {df['City'].nunique()}  
    **Models**: {len(MODEL_CONFIGS)}  
    **Features**: 8
    """)

# ================================================================
# PAGE 1: TỔNG QUAN
# ================================================================
if page == "Tổng Quan":
    # Header
    st.markdown("""
    <div class="dash-header">
        <h1>🌍 DỰ BÁO THỜI TIẾT CHÂU ÂU</h1>
        <p>Phân tích & dự báo nhiệt độ cho 5 vùng khí hậu với Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Cards
    if models_loaded:
        best_r2_overall = max([
            metrics[target][model]["R2"]
            for target in TARGET_COLUMNS
            for model in MODEL_CONFIGS.keys()
        ])
    else:
        best_r2_overall = 0.0
    
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-nhan">SAMPLES</div>
            <div class="kpi-gia xanh">{len(df):,}</div>
            <div class="kpi-phu">records</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-nhan">CITIES</div>
            <div class="kpi-gia xanhla">{df['City'].nunique()}</div>
            <div class="kpi-phu">regions</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-nhan">FEATURES</div>
            <div class="kpi-gia cam">8</div>
            <div class="kpi-phu">engineered</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-nhan">MODELS</div>
            <div class="kpi-gia tim">{len(MODEL_CONFIGS)}</div>
            <div class="kpi-phu">algorithms</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-nhan">BEST R²</div>
            <div class="kpi-gia do">{best_r2_overall:.4f}</div>
            <div class="kpi-phu">accuracy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Temperature by City
    st.markdown('<div class="tieu-de-muc">Nhiệt Độ Trung Bình Theo Thành Phố</div>', unsafe_allow_html=True)
    fig = plot_temperature_by_city(df)
    st.pyplot(fig)
    
    # Data Sample
    st.markdown('<div class="tieu-de-muc">Mẫu Dữ Liệu</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)
    
    # Map
    st.markdown('<div class="tieu-de-muc">Bản Đồ Châu Âu - 5 Vùng Khí Hậu</div>', unsafe_allow_html=True)
    
    # Create map
    m = folium.Map(location=[50, 10], zoom_start=4, tiles="CartoDB dark_matter")
    
    for city, info in CITIES.items():
        folium.CircleMarker(
            location=[info["lat"], info["lon"]],
            radius=10,
            popup=f"<b>{city}</b><br>{info['khi_hau']}",
            color=info["mau"],
            fill=True,
            fillColor=info["mau"],
            fillOpacity=0.7
        ).add_to(m)
    
    st_folium(m, width=None, height=400)


# ================================================================
# PAGE 2: PHÂN TÍCH DỮ LIỆU
# ================================================================
elif page == "Phân Tích Dữ Liệu":
    st.markdown('<div class="tieu-de-muc">Phân Phối Nhiệt Độ</div>', unsafe_allow_html=True)
    
    target_select = st.selectbox(
        "Chọn target:",
        TARGET_COLUMNS,
        format_func=lambda x: TARGET_NAMES[x]
    )
    
    fig = plot_temperature_distribution(df, target_select)
    st.pyplot(fig)
    
    # Correlation
    st.markdown('<div class="tieu-de-muc">Ma Trận Tương Quan</div>', unsafe_allow_html=True)
    fig = plot_correlation_heatmap(df)
    st.pyplot(fig)
    
    # Seasonal patterns
    st.markdown('<div class="tieu-de-muc">Seasonal Patterns</div>', unsafe_allow_html=True)
    
    city_select = st.selectbox("Chọn thành phố:", sorted(CITIES.keys()))
    fig = plot_seasonal_patterns(df, city_select)
    st.pyplot(fig)


# ================================================================
# PAGE 3: ĐÁNH GIÁ MÔ HÌNH
# ================================================================
elif page == "Đánh Giá Mô Hình":
    if not models_loaded:
        st.error("Models chưa được load. Chạy training script trước!")
        st.stop()
    
    st.markdown('<div class="tieu-de-muc">So Sánh Hiệu Suất Models</div>', unsafe_allow_html=True)
    
    target_eval = st.selectbox(
        "Chọn target để đánh giá:",
        TARGET_COLUMNS,
        format_func=lambda x: TARGET_NAMES[x]
    )
    
    # Model comparison chart
    fig = plot_model_comparison(metrics, target_eval)
    st.pyplot(fig)
    
    # Summary table
    st.markdown('<div class="tieu-de-muc">Bảng Tổng Hợp Metrics</div>', unsafe_allow_html=True)
    summary_df = create_summary_table(metrics, target_eval)
    st.dataframe(
        summary_df.style.highlight_max(subset=["R²"], color="#1f3a1f")
                       .highlight_min(subset=["MAE", "RMSE"], color="#1f3a1f")
                       .format({"R²": "{:.4f}", "MAE": "{:.4f}", "RMSE": "{:.4f}"}),
        use_container_width=True
    )
    
    # Best models per target
    st.markdown('<div class="tieu-de-muc">Best Models</div>', unsafe_allow_html=True)
    
    cols = st.columns(3)
    for i, target in enumerate(TARGET_COLUMNS):
        best_model, best_r2 = get_best_model_for_target(target, metrics)
        display_name = MODEL_CONFIGS[best_model]["display_name"]
        
        with cols[i]:
            st.metric(
                label=TARGET_NAMES[target],
                value=display_name,
                delta=f"R² = {best_r2:.4f}"
            )


# ================================================================
# PAGE 4: DỰ BÁO NHIỆT ĐỘ
# ================================================================
elif page == "Dự Báo Nhiệt Độ":
    if not models_loaded:
        st.error("Models chưa được load. Chạy training script trước!")
        st.stop()
    
    st.markdown('<div class="tieu-de-muc">Nhập Điều Kiện Thời Tiết</div>', unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        # Model selection
        model_list = list(MODEL_CONFIGS.keys())
        default_model = max(
            model_list,
            key=lambda m: metrics["TempMax"][m]["R2"]
        )
        
        model_selected = st.selectbox(
            "Chọn mô hình:",
            model_list,
            format_func=lambda x: MODEL_CONFIGS[x]["display_name"],
            index=model_list.index(default_model)
        )
        
        # Input features
        city_selected = st.selectbox("Thành phố:", sorted(CITIES.keys()))
        month_selected = st.slider("Tháng:", 1, 12, 6)
        humidity = st.slider("Độ ẩm (%):", 20, 100, 70)
        precipitation = st.slider("Lượng mưa (mm):", 0.0, 50.0, 2.0, step=0.5)
        wind_speed = st.slider("Tốc độ gió (km/h):", 0.0, 100.0, 20.0, step=1.0)
        
        # Calculate daylight
        from utils.helpers import calculate_daylight_hours
        day_of_year = pd.Timestamp(2024, month_selected, 15).dayofyear
        daylight = calculate_daylight_hours(
            CITIES[city_selected]["lat"],
            day_of_year
        )
        
        # Get lag values
        try:
            lag1 = lag_metadata.loc[(city_selected, month_selected), "TempLag1"]
            lag3 = lag_metadata.loc[(city_selected, month_selected), "TempLag3"]
        except:
            lag1 = 10.0
            lag3 = 10.0
        
        st.info(f"""
        **{city_selected}** — {CITIES[city_selected]['khi_hau']}
        
        Giờ chiếu sáng tháng {month_selected}: **{daylight:.1f} giờ**
        
        Lag tự động: TempLag1={lag1:.1f}°C, TempLag3={lag3:.1f}°C
        """)
    
    with col_right:
        # Make prediction
        predictions = predict_temperature(
            city=city_selected,
            month=month_selected,
            humidity=humidity,
            precipitation=precipitation,
            wind_speed=wind_speed,
            model_name=model_selected,
            models=models,
            scalers=scalers,
            lag_metadata=lag_metadata
        )
        
        # Get R2 scores
        r2_scores = {
            target: metrics[target][model_selected]["R2"]
            for target in TARGET_COLUMNS
        }
        
        # Display predictions
        st.markdown(f"""
        <div class="khung-dubao">
          <div class="nhan-dubao">Mô hình: <strong>{MODEL_CONFIGS[model_selected]['display_name']}</strong></div>
          <br>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.8rem;margin-top:.5rem">
            <div>
              <div class="nhan-dubao">Tối đa</div>
              <div class="nhiet-do-dubao" style="font-size:2.2rem;color:#ff7b72">{predictions['TempMax']:.1f}°C</div>
              <div class="nhan-dubao">R²={r2_scores['TempMax']:.4f}</div>
            </div>
            <div>
              <div class="nhan-dubao">Trung bình</div>
              <div class="nhiet-do-dubao" style="font-size:2.2rem;color:#58a6ff">{predictions['TempMean']:.1f}°C</div>
              <div class="nhan-dubao">R²={r2_scores['TempMean']:.4f}</div>
            </div>
            <div>
              <div class="nhan-dubao">Tối thiểu</div>
              <div class="nhiet-do-dubao" style="font-size:2.2rem;color:#d2a8ff">{predictions['TempMin']:.1f}°C</div>
              <div class="nhan-dubao">R²={r2_scores['TempMin']:.4f}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Historical comparison
        st.markdown('<div class="tieu-de-muc" style="margin-top:1.2rem">So Sánh Với Lịch Sử</div>',
                    unsafe_allow_html=True)
        
        historical_data = {
            "TempMax": df[(df["City"]==city_selected) & (df["Month"]==month_selected)]["TempMax"].mean(),
            "TempMean": df[(df["City"]==city_selected) & (df["Month"]==month_selected)]["TempMean"].mean(),
            "TempMin": df[(df["City"]==city_selected) & (df["Month"]==month_selected)]["TempMin"].mean()
        }
        
        fig = plot_prediction_comparison(
            historical_data, predictions,
            city_selected, month_selected
        )
        st.pyplot(fig)
    
    # Feature importance (if tree-based model)
    if model_selected in ["Random Forest", "XGBoost"]:
        st.markdown('<div class="tieu-de-muc">Feature Importance</div>', unsafe_allow_html=True)
        
        from config import FEATURE_COLUMNS
        model_obj = models["TempMax"][model_selected]
        
        fig = plot_feature_importance(
            model_obj, FEATURE_COLUMNS,
            MODEL_CONFIGS[model_selected]["display_name"],
            "TempMax"
        )
        if fig:
            st.pyplot(fig)


# ================================================================
# FOOTER
# ================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8b949e; font-size: 0.85rem;">
    🌍 Weather ML Project | Data từ Open-Meteo API | 
    Models: Ridge, KNN, Random Forest, XGBoost, SVR
</div>
""", unsafe_allow_html=True)

