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
    print(" PHÂN TÍCH DỮ LIỆU BẰNG NGÔN NGỮ SQL (SQL EDA) ".center(80, "="))
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
        print("\n[TRUY VẤN 1] THỐNG KÊ ĐẶC TRƯNG KHÍ HẬU THEO TỪNG VÙNG:")
        query1 = """
        SELECT 
            City,
            ROUND(AVG(TempMean), 2) AS AvgTemp,
            ROUND(MAX(TempMax), 2) AS RecordHigh,
            ROUND(MIN(TempMin), 2) AS RecordLow,
            ROUND(MAX(TempMean) - MIN(TempMean), 2) AS ThermalRange -- Thay cho Volatility
        FROM weather_stats
        GROUP BY City
        ORDER BY AvgTemp DESC;
        """
        res1 = pd.read_sql_query(query1, conn)
        print(res1.to_string(index=False))

        # --- TRUY VẤN 2: PHÂN TÍCH XU HƯỚNG THEO THÁNG ---
        # Mục đích: Kiểm chứng tính mùa vụ (Seasonality) cho Feature Engineering
        print("\n[TRUY VẤN 2] THỐNG KÊ TẦN SUẤT CÁC NGÀY CỰC ĐOAN")
        query2 = """
        SELECT 
            City,
            SUM(CASE WHEN TempMax > 30 THEN 1 ELSE 0 END) AS HeatwaveDays,
            SUM(CASE WHEN TempMin < 0 THEN 1 ELSE 0 END) AS FrostDays,
            ROUND(SUM(CASE WHEN TempMax > 30 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS HeatRatioPct
        FROM weather_stats
        GROUP BY City
        ORDER BY HeatwaveDays DESC;
        """
        res2 = pd.read_sql_query(query2, conn)
        print(res2.to_string(index=False))

        # --- TRUY VẤN 3: PHÁT HIỆN CÁC ĐIỂM DỮ LIỆU CỰC ĐOAN (ANOMALIES) ---
        # Mục đích: Tìm các ngày có thời tiết khắc nghiệt để kiểm tra độ bền của Model
        print("\n[TRUY VẤN 3] KIỂM TRA TƯƠNG QUAN ĐỘ ẨM VÀ NHIỆT ĐỘ TỐI ĐA:")
        query3 = """
        SELECT 
            City,
            CASE 
                WHEN Humidity > 80 THEN 'High Humidity (>80%)'
                WHEN Humidity < 40 THEN 'Low Humidity (<40%)'
                ELSE 'Normal'
            END AS HumidityRange,
            ROUND(AVG(TempMax), 2) AS AvgMaxTemp
        FROM weather_stats
        GROUP BY City, HumidityRange
        ORDER BY City, AvgMaxTemp DESC;
        """
        res3 = pd.read_sql_query(query3, conn)
        print(res3.to_string(index=False))

        # Đóng kết nối
        conn.close()
        print("\n" + "="*80)
        print(" HOÀN TẤT PHÂN TÍCH SQL ".center(80, "="))
        print("="*80)

    except Exception as e:
        print(f"Lỗi trong quá trình xử lý SQL: {e}")

if __name__ == "__main__":
    run_sql_eda()