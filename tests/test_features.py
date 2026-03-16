# ================================================================
# UNIT TESTS FOR FEATURE ENGINEERING MODULE
# ================================================================

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.feature_engineering import (
    add_temporal_features, add_daylight_hours,
    add_lag_features, encode_city
)
from utils.helpers import calculate_daylight_hours
from config import CITIES


class TestTemporalFeatures:
    """Test cases cho temporal features."""
    
    def test_add_temporal_features_columns(self):
        """Test add_temporal_features tạo đúng columns."""
        df = pd.DataFrame({
            'Date': pd.to_datetime(['2024-01-15', '2024-06-20'])
        })
        
        df_result = add_temporal_features(df)
        
        # Check new columns exist
        assert 'Year' in df_result.columns
        assert 'Month' in df_result.columns
        assert 'Day' in df_result.columns
        assert 'DayOfYear' in df_result.columns
        assert 'Quarter' in df_result.columns
    
    def test_add_temporal_features_values(self):
        """Test add_temporal_features tính toán đúng giá trị."""
        df = pd.DataFrame({
            'Date': pd.to_datetime(['2024-03-15'])
        })
        
        df_result = add_temporal_features(df)
        
        assert df_result['Year'].iloc[0] == 2024
        assert df_result['Month'].iloc[0] == 3
        assert df_result['Day'].iloc[0] == 15
        assert df_result['Quarter'].iloc[0] == 1


class TestDaylightHours:
    """Test cases cho daylight hours calculation."""
    
    def test_calculate_daylight_hours_summer(self):
        """Test daylight hours vào mùa hè (nhiều hơn 12h)."""
        # Tháng 6 (summer solstice) ở Stockholm (lat ~59)
        daylight = calculate_daylight_hours(59.3293, 172)  # Day 172 ~ June 21
        
        # Nên có > 17 hours vào mùa hè ở Stockholm
        assert daylight > 17
    
    def test_calculate_daylight_hours_winter(self):
        """Test daylight hours vào mùa đông (ít hơn 12h)."""
        # Tháng 12 (winter solstice) ở Stockholm
        daylight = calculate_daylight_hours(59.3293, 355)  # Day 355 ~ Dec 21
        
        # Nên có < 7 hours vào mùa đông ở Stockholm
        assert daylight < 7
    
    def test_calculate_daylight_hours_equator(self):
        """Test daylight hours gần xích đạo (~12h quanh năm)."""
        # Latitude 0 (equator)
        daylight_summer = calculate_daylight_hours(0, 172)
        daylight_winter = calculate_daylight_hours(0, 355)
        
        # Nên gần 12 hours
        assert abs(daylight_summer - 12) < 0.5
        assert abs(daylight_winter - 12) < 0.5
    
    def test_add_daylight_hours_column(self):
        """Test add_daylight_hours thêm column."""
        df = pd.DataFrame({
            'City': ['Amsterdam', 'Berlin'],
            'Date': pd.to_datetime(['2024-06-15', '2024-12-15']),
            'DayOfYear': [167, 350]
        })
        
        df_result = add_daylight_hours(df)
        
        assert 'DaylightHours' in df_result.columns
        assert len(df_result['DaylightHours']) == 2


class TestLagFeatures:
    """Test cases cho lag features."""
    
    def test_add_lag_features_columns(self):
        """Test add_lag_features tạo đúng columns."""
        df = pd.DataFrame({
            'City': ['Amsterdam'] * 5,
            'Date': pd.date_range('2024-01-01', periods=5),
            'TempMean': [10, 12, 11, 13, 14]
        })
        
        df_result = add_lag_features(df, lag_days=[1, 3])
        
        assert 'TempLag1' in df_result.columns
        assert 'TempLag3' in df_result.columns
    
    def test_add_lag_features_values(self):
        """Test add_lag_features tính toán đúng giá trị."""
        df = pd.DataFrame({
            'City': ['Amsterdam'] * 5,
            'Date': pd.date_range('2024-01-01', periods=5),
            'TempMean': [10.0, 12.0, 11.0, 13.0, 14.0]
        })
        
        df_result = add_lag_features(df, lag_days=[1])
        
        # Row 1's lag1 should be row 0's value (after sorting)
        # Note: NaN will be filled with mean
        assert df_result['TempLag1'].iloc[1] == 10.0
        assert df_result['TempLag1'].iloc[2] == 12.0
    
    def test_add_lag_features_multiple_cities(self):
        """Test add_lag_features hoạt động đúng với nhiều cities."""
        df = pd.DataFrame({
            'City': ['Amsterdam', 'Amsterdam', 'Berlin', 'Berlin'],
            'Date': pd.date_range('2024-01-01', periods=4),
            'TempMean': [10.0, 12.0, 20.0, 22.0]
        })
        
        df_result = add_lag_features(df, lag_days=[1])
        
        # Lag should be computed within each city group
        # Berlin's lag should not use Amsterdam's values
        assert 'TempLag1' in df_result.columns


class TestCityEncoding:
    """Test cases cho city encoding."""
    
    def test_encode_city_alphabetical(self):
        """Test encode_city dùng alphabetical order."""
        df = pd.DataFrame({
            'City': ['Berlin', 'Amsterdam', 'Zurich', 'Athens']
        })
        
        df_result = encode_city(df)
        
        # Amsterdam should be 0 (first alphabetically)
        assert df_result[df_result['City'] == 'Amsterdam']['City_encoded'].iloc[0] == 0
        # Athens should be 1
        assert df_result[df_result['City'] == 'Athens']['City_encoded'].iloc[0] == 1
        # Berlin should be 2
        assert df_result[df_result['City'] == 'Berlin']['City_encoded'].iloc[0] == 2
    
    def test_encode_city_consistent(self):
        """Test encode_city consistent across runs."""
        df1 = pd.DataFrame({'City': ['Amsterdam', 'Berlin', 'Athens']})
        df2 = pd.DataFrame({'City': ['Berlin', 'Amsterdam', 'Athens']})
        
        df1_result = encode_city(df1)
        df2_result = encode_city(df2)
        
        # Same cities should get same encoding
        amsterdam1 = df1_result[df1_result['City'] == 'Amsterdam']['City_encoded'].iloc[0]
        amsterdam2 = df2_result[df2_result['City'] == 'Amsterdam']['City_encoded'].iloc[0]
        
        assert amsterdam1 == amsterdam2


class TestFeatureValidation:
    """Test cases cho feature validation."""
    
    def test_daylight_hours_range(self):
        """Test daylight hours luôn trong phạm vi hợp lý."""
        for lat in [40, 50, 60]:  # European latitudes
            for day in [1, 100, 200, 300]:
                daylight = calculate_daylight_hours(lat, day)
                # Should be between 0 and 24
                assert 0 <= daylight <= 24
    
    def test_city_config_completeness(self):
        """Test tất cả cities có đầy đủ thông tin."""
        required_keys = ['lat', 'lon', 'khi_hau', 'mau']
        
        for city, info in CITIES.items():
            for key in required_keys:
                assert key in info, f"{city} thiếu key: {key}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
