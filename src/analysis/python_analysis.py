import pandas as pd
import numpy as np

def get_descriptive_stats(df):
    """Thống kê mô tả bậc cao sử dụng Pandas built-in."""
    stats_df = df.describe().T
    # Pandas mặc định sử dụng công thức hiệu chỉnh cho Skew và Kurtosis
    stats_df['skewness'] = df.skew(numeric_only=True)
    stats_df['kurtosis'] = df.kurt(numeric_only=True)
    return stats_df

def check_normality_jb(df, column):
    """
    Kiểm định tính chuẩn bằng phương pháp Jarque-Bera (thuần Numpy).
    Dựa trên độ lệch (Skewness) và độ nhọn (Kurtosis).
    """
    data = df[column].dropna().values
    n = len(data)
    if n < 2: return 0, 1.0
    
    # Tính toán thủ công để chứng minh năng lực toán học
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    s = np.sum((data - mean)**3) / (n * std**3) # Skewness
    k = np.sum((data - mean)**4) / (n * std**4) # Kurtosis
    
    # Công thức Jarque-Bera: JB = (n/6) * [S^2 + (1/4)*(K-3)^2]
    jb_stat = (n / 6.0) * (s**2 + (1/4.0) * (k - 3)**2)
    
    # Với n lớn, JB tuân theo phân phối Chi-square (df=2)
    # Ở đây ta trả về statistic để quan sát độ lệch so với 0
    return jb_stat, s, k

def calculate_correlation(df, method='pearson'):
    """Tính toán ma trận tương quan."""
    return df.corr(method=method, numeric_only=True)

def analyze_city_variance_manual(df, target_col='TempMean'):
    """
    Phân tích phương sai (ANOVA) thủ công bằng Numpy.
    F = MS_between / MS_within
    """
    # 1. Chuẩn bị dữ liệu các nhóm
    groups = [group[target_col].values for name, group in df.groupby('City')]
    all_data = np.concatenate(groups)
    
    # 2. Tính toán các đại lượng tổng
    grand_mean = np.mean(all_data)
    total_n = len(all_data)
    k_groups = len(groups)
    
    # 3. SS_between (Phương sai giữa các nhóm)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    df_between = k_groups - 1
    ms_between = ss_between / df_between
    
    # 4. SS_within (Phương sai nội bộ nhóm)
    ss_within = sum(np.sum((g - np.mean(g))**2) for g in groups)
    df_within = total_n - k_groups
    ms_within = ss_within / df_within
    
    # 5. F-statistic
    f_stat = ms_between / ms_within
    
    return f_stat, df_between, df_within