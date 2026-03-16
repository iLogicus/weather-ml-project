# ================================================================
# UNIT TESTS FOR DATA MODULE
# ================================================================

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.load_data import validate_data, get_data_summary
from config import CITIES


class TestDataLoading:
    """Test cases cho data loading functions."""
    
    def test_validate_data_removes_nulls(self):
        """Test validate_data xóa null values."""
        # Create sample data with nulls
        df = pd.DataFrame({
            'City': ['Amsterdam', 'Berlin', None],
            'TempMax': [20.0, 25.0, 30.0],
            'Humidity': [70, None, 80]
        })
        
        df_validated = validate_data(df)
        
        # Should have no null values
        assert df_validated.isnull().sum().sum() == 0
        # Should have fewer rows
        assert len(df_validated) < len(df)
    
    def test_validate_data_removes_duplicates(self):
        """Test validate_data xóa duplicates."""
        df = pd.DataFrame({
            'City': ['Amsterdam', 'Amsterdam', 'Berlin'],
            'Date': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-02']),
            'TempMax': [20.0, 20.0, 25.0],
            'TempMean': [15.0, 15.0, 20.0],
            'TempMin': [10.0, 10.0, 15.0],
            'Humidity': [70, 70, 75],
            'Precipitation': [0, 0, 5],
            'WindSpeed': [10, 10, 15]
        })
        
        df_validated = validate_data(df)
        
        # Should have no duplicates
        assert df_validated.duplicated().sum() == 0
    
    def test_validate_data_range_checking(self):
        """Test validate_data kiểm tra phạm vi giá trị."""
        df = pd.DataFrame({
            'City': ['Amsterdam', 'Berlin', 'Athens'],
            'Date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'TempMax': [20.0, 150.0, 30.0],  # 150 out of range
            'TempMean': [15.0, 145.0, 25.0],
            'TempMin': [10.0, 140.0, 20.0],
            'Humidity': [70, 75, 80],
            'Precipitation': [0, 5, 10],
            'WindSpeed': [10, 15, 20]
        })
        
        df_validated = validate_data(df)
        
        # Invalid temperature row should be removed
        assert len(df_validated) == 2
        assert 'Berlin' not in df_validated['City'].values


class TestDataSummary:
    """Test cases cho data summary functions."""
    
    def test_get_data_summary_structure(self):
        """Test get_data_summary trả về đúng structure."""
        df = pd.DataFrame({
            'City': ['Amsterdam', 'Berlin'],
            'Date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'TempMax': [20.0, 25.0],
            'TempMean': [15.0, 20.0],
            'TempMin': [10.0, 15.0],
            'Humidity': [70, 75],
            'Precipitation': [0, 5],
            'WindSpeed': [10, 15]
        })
        
        summary = get_data_summary(df)
        
        # Check required keys
        assert 'total_records' in summary
        assert 'cities' in summary
        assert 'num_cities' in summary
        assert 'date_range' in summary
        assert 'temp_stats' in summary
        
        # Check values
        assert summary['total_records'] == 2
        assert summary['num_cities'] == 2
        assert len(summary['cities']) == 2
    
    def test_get_data_summary_stats(self):
        """Test get_data_summary tính toán stats đúng."""
        df = pd.DataFrame({
            'City': ['Amsterdam'] * 3,
            'Date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'TempMax': [20.0, 22.0, 24.0],
            'TempMean': [15.0, 17.0, 19.0],
            'TempMin': [10.0, 12.0, 14.0],
            'Humidity': [70, 75, 80],
            'Precipitation': [0, 5, 10],
            'WindSpeed': [10, 15, 20]
        })
        
        summary = get_data_summary(df)
        
        # Check temp stats
        assert summary['temp_stats']['max']['min'] == 20.0
        assert summary['temp_stats']['max']['max'] == 24.0
        assert summary['temp_stats']['max']['mean'] == pytest.approx(22.0)


class TestDataValidation:
    """Test cases cho data validation logic."""
    
    def test_city_names_valid(self):
        """Test tất cả city names trong config đều valid."""
        for city in CITIES.keys():
            assert isinstance(city, str)
            assert len(city) > 0
    
    def test_city_coordinates_valid(self):
        """Test coordinates trong config đều valid."""
        for city, info in CITIES.items():
            # Latitude: -90 to 90
            assert -90 <= info['lat'] <= 90
            # Longitude: -180 to 180
            assert -180 <= info['lon'] <= 180
    
    def test_european_cities_latitude(self):
        """Test tất cả cities đều ở châu Âu (latitude > 35)."""
        for city, info in CITIES.items():
            assert info['lat'] > 35, f"{city} không ở châu Âu"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
