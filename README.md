# DỰ ÁN DỰ BÁO THỜI TIẾT CHÂU ÂU - ADY201M

Dự án xây dựng hệ thống học máy (Machine Learning) toàn diện nhằm dự báo các chỉ số nhiệt độ cho 5 khu vực khí hậu đặc trưng tại Châu Âu. Hệ thống được thiết kế theo kiến trúc module hóa, đảm bảo tính mở rộng, dễ bảo trì và sẵn sàng cho việc triển khai thực tế.

---

## 1. GIỚI THIỆU TỔNG QUAN

Hệ thống thực hiện dự báo 3 biến mục tiêu: Nhiệt độ tối đa (TempMax), Nhiệt độ trung bình (TempMean) và Nhiệt độ tối thiểu (TempMin). Dữ liệu được thu thập từ 5 thành phố đại diện cho các vùng khí hậu khác nhau:

| Thành phố | Quốc gia | Loại hình khí hậu |
|-----------|----------|-------------------|
| Amsterdam | Hà Lan | Ôn đới hải dương |
| Berlin | Đức | Lục địa |
| Athens | Hy Lạp | Địa Trung Hải |
| Stockholm | Thụy Điển | Bắc Âu |
| Zurich | Thụy Sĩ | Núi cao |

---

## 2. CẤU TRÚC THƯ MỤC DỰ ÁN

Dự án tuân thủ cấu trúc phân tầng chuyên nghiệp để quản lý mã nguồn và dữ liệu tách biệt:

```text
weather-ml-project/
├── app/                          # Giao diện người dùng
│   └── app.py                    # Ứng dụng chính Streamlit
├── data/                         # Quản lý dữ liệu
│   ├── raw/                      # Dữ liệu gốc từ API
│   └── processed/                # Dữ liệu sau khi xử lý đặc trưng
├── models/                       # Quản lý mô hình
│   └── saved_models/             # Tập tin mô hình (.pkl) và metadata
├── src/                          # Mã nguồn cốt lõi
│   ├── data/                     # Xử lý và xác thực dữ liệu
│   ├── features/                 # Kỹ thuật đặc trưng (Feature Engineering)
│   ├── models/                   # Huấn luyện và dự báo
│   ├── utils/                    # Các hàm bổ trợ
│   ├── visualization/            # Trực quan hóa dữ liệu
│   └── config.py                 # Cấu hình toàn hệ thống
├── tests/                        # Hệ thống test tự độnCg
├── run_pipeline.py               # Chạy quy trình tự động
├── setup.py                      # Thiết lập môi trường
└── requirements.txt              # Danh sách thư viện phụ thuộc
```

---

## 3. QUY TRÌNH THIẾT LẬP VÀ VẬN HÀNH

### Bước 1: Khởi tạo môi trường
Yêu cầu hệ thống cài đặt Python phiên bản 3.10 trở lên.

```bash
# Truy cập thư mục dự án
cd weather-ml-project

# Khởi tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường (Windows)
venv\Scripts\activate

# Kích hoạt môi trường (Linux/Mac)
source venv/bin/activate
```

### Bước 2: Cài đặt và cấu hình tự động
Sử dụng tập tin thiết lập để tự động tạo cấu trúc thư mục và cài đặt thư viện:

```bash
python setup.py
```

### Bước 3: Thực thi quy trình dữ liệu và huấn luyện
Chạy quy trình khép kín từ việc tải dữ liệu mới, xử lý đặc trưng đến huấn luyện 15 mô hình:

```bash
python run_pipeline.py --fetch-new
```

### Bước 4: Khởi chạy ứng dụng web
Sau khi hoàn tất huấn luyện, khởi chạy giao diện tương tác:

```bash
streamlit run app/app.py
```

---

## 4. KỸ THUẬT ĐẶC TRƯNG VÀ MÔ HÌNH HÓA

### Các đặc trưng đầu vào (8 đặc trưng)
Hệ thống trích xuất các thông tin quan trọng để tăng độ chính xác cho dự báo:
*   Độ ẩm, Lượng mưa, Tốc độ gió: Dữ liệu thời tiết cơ bản.
*   Tháng và Quý: Đại diện cho tính chu kỳ thời gian.
*   Giờ chiếu sáng (DaylightHours): Tính toán theo toán học thiên văn dựa trên vĩ độ.
*   Nhiệt độ trễ (Lag features): Sử dụng nhiệt độ 1 ngày và 3 ngày trước đó.

### Danh sách thuật toán huấn luyện
Hệ thống sử dụng tổ hợp 5 thuật toán cho mỗi mục tiêu dự báo:
1.  Ridge Regression (Mô hình tuyến tính)
2.  K-Nearest Neighbors (Dự báo dựa trên lân cận)
3.  Random Forest (Học máy dựa trên tập hợp cây)
4.  XGBoost (Gradient Boosting hiệu suất cao)
5.  Support Vector Regression (Hồi quy vector hỗ trợ)

---

## 5. TÍNH NĂNG CỦA ỨNG DỤNG GIAO DIỆN

Ứng dụng Streamlit được chia thành các phân khu chức năng chính:
*   Trang Tổng quan: Hiển thị bản đồ tương tác và thống kê chung về các vùng khí hậu.
*   Trang Phân tích: Trực quan hóa xu hướng nhiệt độ và ma trận tương quan giữa các biến.
*   Trang Đánh giá: So sánh hiệu suất (R2, MAE) giữa các thuật toán để chọn ra mô hình tốt nhất.
*   Trang Dự báo: Cho phép người dùng nhập thông số thực tế để nhận dự báo nhiệt độ tức thì.

---

## 6. KIỂM THỬ VÀ ĐẢM BẢO CHẤT LƯỢNG

Dự án tích hợp hệ thống kiểm thử tự động bằng `pytest` để đảm bảo tính ổn định:
*   Kiểm tra tính toàn vẹn của dữ liệu đầu vào và đầu ra.
*   Xác minh các hàm tính toán đặc trưng (giờ chiếu sáng, lag features).

Cách chạy kiểm thử:
```bash
pytest tests/ -v
```

---

## 7. TÀI LIỆU THAM KHẢO

*   Nguồn dữ liệu: Open-Meteo Archive API.
*   Thư viện chính: Scikit-learn, XGBoost, Streamlit, Pandas, NumPy, Matplotlib, Seaborn, Folium.
*   Tiêu chuẩn mã nguồn: Tuân thủ quy tắc PEP 8 và gợi ý kiểu dữ liệu (Type hinting).

---
**Dự án được thực hiện bởi Nhóm 1 - Môn ADY201m - Kỳ Spring 2026 - FPTU (HCM).**