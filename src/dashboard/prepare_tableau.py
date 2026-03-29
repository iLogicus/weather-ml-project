# ================================================================
# TABLEAU DATA PREPARATION
# Tự động chọn mô hình tốt nhất và tạo Master Dataset cho Tableau
# ================================================================

import pandas as pd
import numpy as np
import os
import sys

# Thêm đường dẫn thư mục gốc để import các module từ src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FEATURE_COLUMNS, TARGET_COLUMNS, MODELS_DIR
from utils.helpers import print_section_header

# Import các thành phần dự báo từ các module con đã viết
try:
    from features.feature_engineering import load_processed_data
    from models.predict import load_all_models, load_scalers, load_evaluation_metrics, get_best_model_for_target
except ImportError as e:
    print(f"Lỗi Import: {e}. Vui lòng kiểm tra lại cấu trúc thư mục src.")
    sys.exit(1)

def generate_tableau_dataset():
    """
    Sử dụng mô hình tốt nhất để tạo dataset toàn diện lưu vào data/external/
    """
    print_section_header("KHỞI TẠO DỮ LIỆU DASHBOARD TABLEAU")
    
    # 1. Load các tài nguyên AI
    print("1. Đang nạp Models, Scalers và Metrics...")
    df = load_processed_data()
    models = load_all_models()
    scalers = load_scalers()
    metrics = load_evaluation_metrics()
    
    if not all([models, scalers, metrics]):
        print("Cảnh báo: Thiếu file huấn luyện. Hãy chạy train_model.py trước.")
        return None
        
    df_export = df.copy()
    
    # 2. Thực hiện dự báo hàng loạt với mô hình tốt nhất cho mỗi target
    print("\n2. Đang thực hiện Batch Prediction bằng mô hình tốt nhất...")
    for target in TARGET_COLUMNS:
        best_model_name, best_r2 = get_best_model_for_target(target, metrics)
        print(f"   -> Target [{target}]: Best Model là {best_model_name} (R²={best_r2:.4f})")
        
        # Lấy feature matrix
        X = df[FEATURE_COLUMNS].values
        # Scale theo đúng scaler của target đó
        X_scaled = scalers[target].transform(X)
        
        # Predict bằng model tốt nhất của target đó
        y_pred = models[target][best_model_name].predict(X_scaled)
        
        # Tạo các cột báo cáo chuyên sâu cho Tableau
        df_export[f'Pred_{target}'] = np.round(y_pred, 1)
        df_export[f'Error_{target}'] = np.round(df_export[f'Pred_{target}'] - df_export[target], 2)
        df_export[f'AbsError_{target}'] = df_export[f'Error_{target}'].abs()
        df_export[f'ModelUsed_{target}'] = best_model_name

    # 3. Thiết lập đường dẫn lưu trữ: data/external/
    # Nhảy ngược 2 cấp từ src/dashboard để về root
    current_file_path = os.path.abspath(__file__) # src/dashboard/prepare_tableau.py
    dashboard_dir = os.path.dirname(current_file_path) # src/dashboard
    src_dir = os.path.dirname(dashboard_dir) # src
    root_dir = os.path.dirname(src_dir) # root_project
    
    output_dir = os.path.join(root_dir, "data", "external")
    
    # Tự động tạo thư mục external nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "tableau_master_dataset.csv")
    
    # 4. Xuất file CSV
    df_export.to_csv(output_path, index=False)
    
    print_section_header("DỮ LIỆU TABLEAU ĐÃ SẴN SÀNG")
    
    return output_path

if __name__ == "__main__":
    generate_tableau_dataset()