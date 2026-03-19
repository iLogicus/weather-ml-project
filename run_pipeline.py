#!/usr/bin/env python3
# ================================================================
# AUTOMATED PIPELINE SCRIPT
# Chạy toàn bộ pipeline từ fetch data → SQL EDA → train models
# ================================================================

import sys
import os
import argparse
from datetime import datetime

# Đảm bảo Python có thể tìm thấy thư mục 'src'
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import các module từ thư mục src
from data.load_data import load_raw_data, validate_data, print_data_info
from data.sql_analysis import run_sql_eda  # <-- Import module SQL mới thêm
from features.feature_engineering import (
    create_all_features, calculate_lag_statistics,
    save_processed_data
)
from models.train_models import train_all_models, save_all_models, print_evaluation_summary
from utils.helpers import save_pickle, print_section_header
from config import MODELS_DIR

def main(fetch_new_data=False, skip_training=False, skip_sql=False):
    """
    Chạy toàn bộ ML pipeline (End-to-End).
    
    Parameters:
    -----------
    fetch_new_data : bool
        True = Fetch dữ liệu mới nhất từ Open-Meteo API.
    skip_training : bool
        True = Dừng lại sau khi hoàn thành Feature Engineering và SQL EDA.
    skip_sql : bool
        True = Bỏ qua bước truy vấn phân tích dữ liệu bằng SQLite3.
    """
    start_time = datetime.now()
    
    print("="*80)
    print("WEATHER ML PIPELINE (INTEGRATED WITH SQL EDA)".center(80))
    print("="*80)
    print(f"\nBắt đầu chạy lúc: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Cấu hình: Fetch Data ({fetch_new_data}), Skip Train ({skip_training}), Skip SQL ({skip_sql})\n")
    
    try:
        # ================================================================
        # BƯỚC 1: LOAD VÀ VALIDATE DATA (TỪ API HOẶC LOCAL)
        # ================================================================
        print_section_header("BƯỚC 1: NẠP VÀ XÁC THỰC DỮ LIỆU", 80)
        
        df_raw = load_raw_data(fetch_new=fetch_new_data)
        df_raw = validate_data(df_raw)
        print_data_info(df_raw)
        
        # ================================================================
        # BƯỚC 2: FEATURE ENGINEERING (TẠO ĐẶC TRƯNG MỚI)
        # ================================================================
        print_section_header("BƯỚC 2: KỸ THUẬT ĐẶC TRƯNG (FEATURE ENGINEERING)", 80)
        
        df_processed = create_all_features(df_raw)
        save_processed_data(df_processed)
        
        # Tính toán và lưu trữ siêu dữ liệu Lag (Dùng cho dự báo thời gian thực)
        lag_stats = calculate_lag_statistics(df_processed)
        lag_stats_path = os.path.join(MODELS_DIR, "lag_metadata.pkl")
        save_pickle(lag_stats, lag_stats_path)
        
        print(f"\n\Hoàn thành Kỹ thuật Đặc trưng!")
        print(f"   Kích thước tập dữ liệu cuối cùng: {df_processed.shape}")
        
        # ================================================================
        # BƯỚC 2.5: SQL EXPLORATORY DATA ANALYSIS (EDA)
        # ================================================================
        if not skip_sql:
            # Gọi hàm phân tích dữ liệu bằng ngôn ngữ SQL
            run_sql_eda()
        else:
            print("\n⏭ Bỏ qua bước phân tích SQL (--no-sql flag)")
        
        # ================================================================
        # KIỂM TRA ĐIỀU KIỆN DỪNG SỚM
        # ================================================================
        if skip_training:
            print("\n⏭ Dừng quy trình trước khi huấn luyện mô hình (--skip-training flag)")
            print(f"Thời gian thực thi: {datetime.now() - start_time}")
            return
        
        # ================================================================
        # BƯỚC 3: HUẤN LUYỆN ĐA MÔ HÌNH (MULTI-MODEL TRAINING)
        # ================================================================
        print_section_header("BƯỚC 3: HUẤN LUYỆN 15 MÔ HÌNH HỌC MÁY", 80)
        
        all_results = train_all_models(df_processed)
        
        # ================================================================
        # BƯỚC 4: LƯU TRỮ VÀ TỔNG HỢP KẾT QUẢ
        # ================================================================
        print_evaluation_summary(all_results)
        save_all_models(all_results)
        
        # ================================================================
        # TỔNG KẾT QUÁ TRÌNH
        # ================================================================
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*80)
        print(" HOÀN TẤT QUY TRÌNH (PIPELINE COMPLETE) ".center(80, "="))
        print("="*80)
        print(f"\nKết thúc lúc: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Tổng thời gian chạy: {duration}")
        print(f"\nThống kê dữ liệu: {len(df_processed):,} bản ghi đã xử lý.")
        print(f"Số lượng mô hình: 3 biến mục tiêu × 5 thuật toán = 15 mô hình.")
        print(f"Thư mục lưu trữ: {MODELS_DIR}")
        print(f"\nĐã sẵn sàng khởi chạy UI: streamlit run app/app.py")
        print("="*80)
        
    except Exception as e:
        print(f"\nLỖI NGHIÊM TRỌNG TRONG QUÁ TRÌNH CHẠY: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Thiết lập bộ phân tích tham số dòng lệnh (Argparse)
    parser = argparse.ArgumentParser(
        description="Khởi chạy toàn bộ quy trình Weather ML Pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--fetch-new",
        action="store_true",
        help="Tải dữ liệu mới nhất từ Open-Meteo API (Mặc định: Dùng dữ liệu local đã lưu)"
    )
    
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Bỏ qua bước huấn luyện mô hình (Chỉ chạy xử lý dữ liệu và SQL)"
    )

    parser.add_argument(
        "--no-sql",
        action="store_true",
        help="Bỏ qua bước phân tích dữ liệu bằng SQL (Dành cho việc muốn train nhanh)"
    )
    
    args = parser.parse_args()
    
    # Kích hoạt hàm main với các tham số từ người dùng
    main(
        fetch_new_data=args.fetch_new,
        skip_training=args.skip_training,
        skip_sql=args.no_sql
    )