# ================================================================
# SQL-BASED EXPLORATORY DATA ANALYSIS (EDA)
# Sử dụng SQLite3 để truy vấn và phân tích dữ liệu thời tiết
# ================================================================

import sqlite3
import pandas as pd
import os
import sys

# Thêm đường dẫn để đảm bảo import được config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_FILE

def run_sql_eda():
    """
    Thực hiện phân tích dữ liệu khám phá (EDA) bằng ngôn ngữ SQL.
    Hàm này sẽ nạp dữ liệu từ CSV vào một cơ sở dữ liệu SQLite trong bộ nhớ (In-memory).
    """
    print("\n" + "="*80)
    print(" BƯỚC 2.5: PHÂN TÍCH DỮ LIỆU BẰNG NGÔN NGỮ SQL (SQL EDA) ".center(80, "="))
    print("="*80)

    # 1. Kiểm tra sự tồn tại của file dữ liệu đã xử lý
    if not os.path.exists(PROCESSED_DATA_FILE):
        print(f"Lỗi: Không tìm thấy tệp {PROCESSED_DATA_FILE}")
        print("Hãy đảm bảo bước Feature Engineering đã hoàn tất.")
        return

    try:
        # 2. Đọc dữ liệu từ file CSV đã có features
        df = pd.read_csv(PROCESSED_DATA_FILE)
        
        # 3. Khởi tạo kết nối SQLite (Sử dụng bộ nhớ RAM để tối ưu tốc độ)
        conn = sqlite3.connect(':memory:')
        
        # 4. Đẩy dữ liệu vào bảng SQL có tên 'weather_stats'
        df.to_sql('weather_stats', conn, index=False, if_exists='replace')
        
        print(f"Đã nạp {len(df)} bản ghi vào SQLite Database.")

        # --- TRUY VẤN 1: THỐNG KÊ TỔNG QUAN THEO TỪNG THÀNH PHỐ ---
        # Mục đích: Hiểu sự khác biệt về khí hậu giữa các vùng (Oceanic, Mediterranean, etc.)
        print("\n[TRUY VẤN 1] THỐNG KÊ NHIỆT ĐỘ TRUNG BÌNH THEO TỪNG VÙNG KHÍ HẬU:")
        query1 = """
        SELECT 
            City AS Thanh_Pho,
            ROUND(AVG(TempMean), 2) AS Nhiet_Do_TB,
            MAX(TempMax) AS Max_Nhiet_Do,
            MIN(TempMin) AS Min_Nhiet_Do,
            ROUND(AVG(Humidity), 1) AS Do_Am_TB
        FROM weather_stats
        GROUP BY City
        ORDER BY Nhiet_Do_TB DESC;
        """
        res1 = pd.read_sql_query(query1, conn)
        print(res1.to_string(index=False))

        # --- TRUY VẤN 2: PHÂN TÍCH XU HƯỚNG THEO THÁNG ---
        # Mục đích: Kiểm chứng tính mùa vụ (Seasonality) cho Feature Engineering
        print("\n[TRUY VẤN 2] BIẾN THIÊN NHIỆT ĐỘ VÀ GIỜ CHIẾU SÁNG THEO THÁNG:")
        query2 = """
        SELECT 
            Month AS Thang,
            ROUND(AVG(TempMax), 2) AS Max_TB,
            ROUND(AVG(DaylightHours), 2) AS Gio_Chieu_Sang_TB,
            ROUND(AVG(Precipitation), 2) AS Luong_Mua_TB
        FROM weather_stats
        GROUP BY Month
        ORDER BY Month ASC;
        """
        res2 = pd.read_sql_query(query2, conn)
        print(res2.to_string(index=False))

        # --- TRUY VẤN 3: PHÁT HIỆN CÁC ĐIỂM DỮ LIỆU CỰC ĐOAN (ANOMALIES) ---
        # Mục đích: Tìm các ngày có thời tiết khắc nghiệt để kiểm tra độ bền của Model
        print("\n[TRUY VẤN 3] DANH SÁCH CÁC NGÀY CÓ LƯỢNG MƯA VÀ GIÓ MẠNH ĐỘT BIẾN:")
        query3 = """
        SELECT Date, City, Precipitation, WindSpeed
        FROM weather_stats
        WHERE Precipitation > 45 OR WindSpeed > 60
        ORDER BY Precipitation DESC
        LIMIT 8;
        """
        res3 = pd.read_sql_query(query3, conn)
        if not res3.empty:
            print(res3.to_string(index=False))
        else:
            print("   (Không tìm thấy ngày nào có điều kiện cực đoan vượt ngưỡng)")

        # --- TRUY VẤN 4: KIỂM TRA TƯƠNG QUAN GIỮA GIỜ NẮNG VÀ NHIỆT ĐỘ ---
        # Mục đích: Chứng minh đặc trưng DaylightHours (tính bằng toán học) có giá trị dự báo
        print("\n[TRUY VẤN 4] KIỂM TRA MỐI QUAN HỆ GIỮA GIỜ CHIẾU SÁNG VÀ NHIỆT ĐỘ:")
        query4 = """
        SELECT 
            CASE 
                WHEN DaylightHours < 10 THEN 'Ngan (<10h)'
                WHEN DaylightHours BETWEEN 10 AND 14 THEN 'Trung Binh (10-14h)'
                ELSE 'Dai (>14h)'
            END AS Nhom_Gio_Nang,
            ROUND(AVG(TempMean), 2) AS Nhiet_Do_TB
        FROM weather_stats
        GROUP BY Nhom_Gio_Nang;
        """
        res4 = pd.read_sql_query(query4, conn)
        print(res4.to_string(index=False))

        # Đóng kết nối
        conn.close()
        print("\n" + "="*80)
        print(" HOÀN TẤT PHÂN TÍCH SQL ".center(80, "="))
        print("="*80)

    except Exception as e:
        print(f"Lỗi trong quá trình xử lý SQL: {e}")

if __name__ == "__main__":
    run_sql_eda()