#!/usr/bin/env python3
# ================================================================
# TỰ ĐỘNG THIẾT LẬP DỰ ÁN (SETUP SCRIPT)
# ================================================================

import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    """In tiêu đề định dạng."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def check_python_version():
    """Kiểm tra phiên bản Python >= 3.10."""
    print_header("KIỂM TRA PHIÊN BẢN PYTHON")
    
    version = sys.version_info
    print(f"Phiên bản Python hiện tại: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("LỖI: Yêu cầu phiên bản Python 3.10 trở lên!")
        return False
    
    print("Kết quả: Phiên bản Python hợp lệ")
    return True


def create_directories():
    """Tạo các thư mục bắt buộc cho dự án."""
    print_header("KHỞI TẠO CÁC THƯ MỤC")
    
    directories = [
        "data/raw",
        "data/processed",
        "data/external",
        "models/saved_models",
        "notebooks",
        "reports/figures",
        "tests"
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"[ĐÃ TẠO]: {directory}")
        else:
            print(f"[ĐÃ TỒN TẠI]: {directory}")


def install_dependencies():
    """Cài đặt các thư viện từ requirements.txt."""
    print_header("CÀI ĐẶT CÁC THƯ VIỆN PHỤ THUỘC")
    
    try:
        print("Đang cài đặt các gói thư viện... Vui lòng chờ giây lát...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"
        ])
        print("Kết quả: Tất cả thư viện đã được cài đặt thành công")
        return True
    except subprocess.CalledProcessError:
        print("LỖI: Quá trình cài đặt thư viện thất bại")
        return False


def verify_imports():
    """Xác minh các thư viện quan trọng đã có thể sử dụng."""
    print_header("XÁC MINH CÁC THƯ VIỆN ĐÃ CÀI ĐẶT")
    
    required_modules = [
        "pandas",
        "numpy",
        "sklearn",
        "xgboost",
        "matplotlib",
        "seaborn",
        "streamlit",
        "folium",
        "requests"
    ]
    
    all_ok = True
    for module in required_modules:
        try:
            __import__(module)
            print(f"[KHỚP] {module}")
        except ImportError:
            print(f"[LỖI] {module} - CHƯA ĐƯỢC CÀI ĐẶT")
            all_ok = False
    
    return all_ok


def check_project_structure():
    """Kiểm tra sự tồn tại của các tệp tin mã nguồn quan trọng."""
    print_header("KIỂM TRA CẤU TRÚC TỆP TIN DỰ ÁN")
    
    required_files = [
        "src/config.py",
        "src/data/load_data.py",
        "src/features/feature_engineering.py",
        "src/models/train_models.py",
        "src/models/predict.py",
        "src/utils/helpers.py",
        "src/visualization/plots.py",
        "app/app.py",
        "requirements.txt",
        "README.md",
        "run_pipeline.py"
    ]
    
    all_ok = True
    for file in required_files:
        path = Path(file)
        if path.exists():
            print(f"[OK] {file}")
        else:
            print(f"[THIẾU] {file}")
            all_ok = False
    
    return all_ok


def print_next_steps():
    """In hướng dẫn các bước tiếp theo."""
    print_header("THIẾT LẬP HOÀN TẤT!")
    
    print("""
Dự án Weather ML của bạn đã sẵn sàng!

HƯỚNG DẪN CHẠY NHANH:

1. Chạy toàn bộ quy trình tự động (Pipeline):
   python run_pipeline.py

2. Chạy thủ công từng phần:
   Huấn luyện: python src/models/train_models.py
   Giao diện:  streamlit run app/app.py

3. Kiểm thử hệ thống:
   pytest tests/ -v

4. Phân tích dữ liệu bằng Notebook:
   jupyter notebook notebooks/exploratory_analysis.ipynb

TÀI LIỆU THAM KHẢO:
   - README.md      : Hướng dẫn chi tiết toàn bộ dự án

HỖ TRỢ KỸ THUẬT:
   - Đảm bảo bạn đang đứng tại thư mục gốc của dự án khi chạy lệnh
   - Kiểm tra kết nối Internet nếu cần tải dữ liệu từ API

Chúc bạn thực hiện dự án thành công!
    """)


def main():
    """Hàm điều khiển thiết lập chính."""
    print("""
+----------------------------------------------------------------+
|                                                                |
|                    DỰ ÁN WEATHER ML - SETUP                    |
|                                                                |
+----------------------------------------------------------------+
    """)
    
    # Đảm bảo làm việc tại thư mục chứa file setup.py
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    steps = [
        ("Kiểm tra phiên bản Python", check_python_version),
        ("Kiểm tra cấu trúc mã nguồn", check_project_structure),
        ("Khởi tạo danh sách thư mục", create_directories),
        ("Cài đặt thư viện bổ trợ", install_dependencies),
        ("Xác minh nạp thư viện", verify_imports)
    ]
    
    results = {}
    for step_name, step_func in steps:
        try:
            if step_name == "Khởi tạo danh sách thư mục":
                step_func()
                results[step_name] = True
            else:
                results[step_name] = step_func()
        except Exception as e:
            print(f"LỖI xảy ra tại bước {step_name}: {e}")
            results[step_name] = False
    
    # In bảng tóm tắt kết quả
    print_header("TÓM TẮT QUÁ TRÌNH THIẾT LẬP")
    for step_name, success in results.items():
        status = "ĐẠT" if success else "THẤT BẠI"
        print(f"{step_name:.<50} {status}")
    
    if all(results.values()):
        print_next_steps()
        return 0
    else:
        print("\nCẢNH BÁO: Một số bước thiết lập đã thất bại. Vui lòng kiểm tra lại.")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nĐã hủy bỏ quá trình thiết lập theo yêu cầu.")
        sys.exit(1)