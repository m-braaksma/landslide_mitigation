"""
Simple practical test for preprocess_nasa_glc function.
Tests that real GLC data produces expected outputs.

To run: 
    pytest test_input_data_tasks_simple.py -v
"""

import os
import sys
import tempfile
import shutil
import random
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import hazelbean as hb
import pygeoprocessing as pygeo

# Import your function
# Adjust this import based on your project structure
from landslide_mitigation_tasks.input_data_tasks import preprocess_nasa_glc


def test_preprocess_nasa_glc_with_real_data():
    """
    End-to-end test with real GLC data:
    1. Load real data and pick a random year
    2. Calculate expected outputs based on inputs
    3. Run the function
    4. Verify outputs match expectations
    """
    
    # ========================================================================
    # SETUP: Create ProjectFlow with same structure as main run file
    # ========================================================================
    p = hb.ProjectFlow()
    p.user_dir = os.path.expanduser('~')
    p.base_data_dir = os.path.join(p.user_dir, 'Files', 'base_data', 'landslide_mitigation')
    
    # Use temp directory for outputs
    temp_dir = tempfile.mkdtemp(prefix='landslide_test_')
    try:
        p.project_dir = temp_dir
        p.preprocess_data_dir = os.path.join(temp_dir, 'preprocessed')
        os.makedirs(p.preprocess_data_dir, exist_ok=True)
        
        # Set parameters (same as main run file)
        p.time_range = range(2000, 2021)
        p.max_location_accuracy_m = 1000
        p.run_this = True
        
        # Get paths to real data
        p.reference_raster_path = p.get_path(
            os.path.join(p.base_data_dir, 'worldpop', 'ppp_2000_1km_Aggregated.tif')
        )
        glc_csv_path = p.get_path(
            os.path.join(p.base_data_dir, 'nasa_glc', 'Global_Landslide_Catalog_Export_rows.csv')
        )
        
        # Get pixel size from reference raster
        ref_info = pygeo.get_raster_info(p.reference_raster_path)
        pixel_width = abs(ref_info['pixel_size'][0])
        pixel_height = abs(ref_info['pixel_size'][1])
        pixel_area_m2 = pixel_width * pixel_height
        
        print(f"\nReference raster pixel size: {pixel_width}m × {pixel_height}m")
        print(f"Pixel area: {pixel_area_m2:,.0f} m²")
        
        # ========================================================================
        # LOAD AND ANALYZE INPUT DATA
        # ========================================================================
        print(f"\nLoading GLC data from: {glc_csv_path}")
        glc_df = pd.read_csv(glc_csv_path)
        print(f"Total events in CSV: {len(glc_df)}")
        
        # Parse dates
        glc_df['event_date'] = pd.to_datetime(
            glc_df['event_date'], 
            format='%m/%d/%Y %I:%M:%S %p', 
            errors='coerce'
        )
        
        # Filter by accuracy (same as the function does)
        accuracy_map = {
            'exact': 0,
            'unknown': np.nan,
            '1km': 1000,
            '5km': 5000,
            '10km': 10000,
            '25km': 25000,
            '50km': 50000
        }
        glc_df['location_accuracy_m'] = glc_df['location_accuracy'].map(accuracy_map)
        glc_df = glc_df[glc_df['location_accuracy_m'] <= p.max_location_accuracy_m]
        print(f"Events after accuracy filter (≤{p.max_location_accuracy_m}m): {len(glc_df)}")
        
        # Get years with events
        glc_df['year'] = glc_df['event_date'].dt.year
        years_with_events = glc_df['year'].dropna().unique()
        years_in_range = [y for y in years_with_events if y in p.time_range]
        
        if len(years_in_range) == 0:
            raise ValueError("No years with events in the specified time range!")
        
        # Pick a random year
        test_year = random.choice(years_in_range)
        print(f"\nRandomly selected year for testing: {test_year}")
        
        # Filter to test year
        yearly_df = glc_df[glc_df['year'] == test_year].copy()
        num_events = len(yearly_df)
        print(f"Number of landslide events in {test_year}: {num_events}")
        
        # Calculate expected pixel count
        # Each event is buffered by its location_accuracy_m
        yearly_df['buffer_area_m2'] = np.pi * (yearly_df['location_accuracy_m'] ** 2)
        yearly_df['expected_pixels'] = yearly_df['buffer_area_m2'] / pixel_area_m2
        total_expected_pixels = yearly_df['expected_pixels'].sum()
        print(f"Expected total pixels (theoretical): {total_expected_pixels:.1f}")
        
        # Calculate expected total mortality
        total_expected_mortality = yearly_df['fatality_count'].fillna(0).sum()
        print(f"Expected total mortality: {total_expected_mortality:.1f}")
        
        # ========================================================================
        # RUN THE FUNCTION
        # ========================================================================
        print(f"\nRunning preprocess_nasa_glc()...")
        preprocess_nasa_glc(p)
        
        # ========================================================================
        # VERIFY OUTPUTS
        # ========================================================================
        binary_raster_path = os.path.join(p.preprocess_data_dir, f'glc_binary_{test_year}.tif')
        mortality_raster_path = os.path.join(p.preprocess_data_dir, f'glc_mortality_{test_year}.tif')
        
        # Check files exist
        assert os.path.exists(binary_raster_path), f"Binary raster not created: {binary_raster_path}"
        assert os.path.exists(mortality_raster_path), f"Mortality raster not created: {mortality_raster_path}"
        print(f"\n✓ Output files created successfully")
        
        # Read binary raster
        with rasterio.open(binary_raster_path) as src:
            binary_arr = src.read(1)
            nodata = src.nodatavals[0]
            actual_pixel_count = np.sum((binary_arr != nodata) & (binary_arr == 1))
        
        print(f"\nBinary raster results:")
        print(f"  Expected pixels (theoretical): {total_expected_pixels:.1f}")
        print(f"  Actual pixels marked as 1: {actual_pixel_count}")
        print(f"  Ratio (actual/expected): {actual_pixel_count/total_expected_pixels:.2f}")
        
        # Check binary pixels within reasonable range (80-120% due to overlaps and ALL_TOUCHED)
        lower_bound = total_expected_pixels * 0.8
        upper_bound = total_expected_pixels * 1.2
        assert lower_bound <= actual_pixel_count <= upper_bound, \
            f"Binary pixel count {actual_pixel_count} outside expected range [{lower_bound:.1f}, {upper_bound:.1f}]"
        print(f"✓ Binary pixel count within acceptable range (80-120% of theoretical)")
        
        # Read mortality raster
        with rasterio.open(mortality_raster_path) as src:
            mortality_arr = src.read(1)
            nodata = src.nodatavals[0]
            # Sum only non-nodata values
            actual_mortality_sum = np.sum(mortality_arr[mortality_arr != nodata])
        
        print(f"\nMortality raster results:")
        print(f"  Expected total mortality: {total_expected_mortality:.2f}")
        print(f"  Actual mortality sum: {actual_mortality_sum:.2f}")
        
        # Check mortality within 0.1% tolerance
        if total_expected_mortality > 0:
            mortality_diff_percent = abs(actual_mortality_sum - total_expected_mortality) / total_expected_mortality * 100
            print(f"  Difference: {mortality_diff_percent:.3f}%")
            assert mortality_diff_percent <= 0.1, \
                f"Mortality sum differs by {mortality_diff_percent:.3f}% (expected ≤0.1%)"
            print(f"✓ Mortality sum within 0.1% of expected")
        else:
            assert actual_mortality_sum == 0, "Expected 0 mortality but got non-zero sum"
            print(f"✓ No mortality events, raster correctly sums to 0")
        
        print(f"\n{'='*60}")
        print(f"✓ ALL TESTS PASSED for year {test_year}")
        print(f"{'='*60}")
        
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\nCleaned up temporary directory")


if __name__ == '__main__':
    """Allow running this test directly for debugging"""
    print("Running test directly...")
    test_preprocess_nasa_glc_with_real_data()
    print("\n✓ Test completed successfully!")