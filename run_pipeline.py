#!/usr/bin/env python3
# ================================================================
# AUTOMATED PIPELINE SCRIPT
# Chạy toàn bộ pipeline từ fetch data → train models
# ================================================================

import sys
import os
import argparse
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.load_data import load_raw_data, validate_data, print_data_info
from features.feature_engineering import (
    create_all_features, calculate_lag_statistics,
    save_processed_data
)
from models.train_models import train_all_models, save_all_models, print_evaluation_summary
from utils.helpers import save_pickle, print_section_header
from config import MODELS_DIR


def main(fetch_new_data=False, skip_training=False):
    """
    Chạy toàn bộ ML pipeline.
    
    Parameters:
    -----------
    fetch_new_data : bool
        True = fetch data mới từ API
    skip_training : bool
        True = chỉ chạy đến feature engineering
    """
    start_time = datetime.now()
    
    print("="*80)
    print("WEATHER ML PIPELINE".center(80))
    print("="*80)
    print(f"\nBắt đầu: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Fetch new data: {fetch_new_data}")
    print(f"Skip training: {skip_training}")
    print()
    
    try:
        # ================================================================
        # STEP 1: LOAD DATA
        # ================================================================
        print_section_header("STEP 1: LOAD DATA", 80)
        
        df_raw = load_raw_data(fetch_new=fetch_new_data)
        df_raw = validate_data(df_raw)
        print_data_info(df_raw)
        
        # ================================================================
        # STEP 2: FEATURE ENGINEERING
        # ================================================================
        print_section_header("STEP 2: FEATURE ENGINEERING", 80)
        
        df_processed = create_all_features(df_raw)
        save_processed_data(df_processed)
        
        # Calculate and save lag statistics
        lag_stats = calculate_lag_statistics(df_processed)
        lag_stats_path = os.path.join(MODELS_DIR, "lag_metadata.pkl")
        save_pickle(lag_stats, lag_stats_path)
        
        print(f"\nFeature engineering complete!")
        print(f"   Final shape: {df_processed.shape}")
        
        if skip_training:
            print("\nSkipping training (--skip-training flag)")
            return
        
        # ================================================================
        # STEP 3: TRAIN MODELS
        # ================================================================
        print_section_header("STEP 3: TRAIN MODELS", 80)
        
        all_results = train_all_models(df_processed)
        
        # ================================================================
        # STEP 4: SAVE MODELS
        # ================================================================
        print_evaluation_summary(all_results)
        save_all_models(all_results)
        
        # ================================================================
        # SUMMARY
        # ================================================================
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE!".center(80))
        print("="*80)
        print(f"\nKết thúc: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Thời gian: {duration}")
        print(f"\nData: {len(df_processed):,} records")
        print(f"Models trained: {len(all_results)} targets × 5 models = 15 models")
        print(f"Models saved to: {MODELS_DIR}")
        print(f"\nReady to run: streamlit run app/app.py")
        print("="*80)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run complete Weather ML pipeline"
    )
    
    parser.add_argument(
        "--fetch-new",
        action="store_true",
        help="Fetch new data from API (default: use cached data)"
    )
    
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training (only run data processing)"
    )
    
    args = parser.parse_args()
    
    main(
        fetch_new_data=args.fetch_new,
        skip_training=args.skip_training
    )
