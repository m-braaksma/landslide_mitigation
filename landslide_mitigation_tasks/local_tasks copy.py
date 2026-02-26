import os
import sys
import hazelbean as hb
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import math
import json
import pickle
import re
import ast
from tqdm import tqdm
from osgeo import gdal, ogr, osr
import pygeoprocessing as pygeo
import rasterio
from rasterio.features import rasterize
import gc
from scipy.ndimage import generic_filter


# import pyarrow as pa
# import pyarrow.parquet as pq

# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# from statsmodels.iolib.summary2 import summary_col

# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker

# Reset GDAL_DATA path after importing hazelbean
# gdal_data_path = os.environ.get("GDAL_DATA")
# print("GDAL_DATA:", gdal_data_path)
# os.environ['GDAL_DATA'] = '/users/0/braak014/miniforge3/envs/teems02/share/gdal'

def s3_file_exists(path):
    ds = gdal.Open(path)
    exists = ds is not None
    ds = None  # Close GDAL dataset
    return exists

def preprocess_aligned_rasters(p):
    """
    Gap-fill and harmonize aligned rasters before tiling.

    Uses WorldPop 2019 as the land mask — it has clean global coverage and
    explicit NDV for ocean/Antarctica, making it the most reliable reference
    for where we expect valid predictions.

    Per-raster fill strategy:
      - avoided_export: river/stream pixels are NDV in InVEST output; fill with 0
        (no vegetation = no avoided export, not unknown)
      - WorldPop: any remaining NDV on land filled with 0
      - ERA5: nearest-neighbor fill for any gaps (ERA5 is rarely missing)
      - gfld: NOT filled — NDV means unknown mortality, exclude from training only

    Writes filled rasters to p.cur_dir for downstream tile tasks.
    """
    if not p.run_this:
        return
    
    years = list(range(2004, 2020))

    # Check if all expected outputs exist; if so, skip everything
    output_dir = os.path.join(p.s3_proj_dir, 'preprocessed_rasters')
    expected_files = []
    for year in years:
        expected_files.extend([
            os.path.join(output_dir, f'avoided_export_{year}_filled.tif'),
            os.path.join(output_dir, f'ppp_{year}_1km_Aggregated_filled.tif'),
            os.path.join(output_dir, f'ERA5_annual_precip_{year}_filled.tif'),
        ])
    if all(s3_file_exists(f) for f in expected_files):
        hb.log('All preprocessed rasters already exist in s3. Skipping preprocess_aligned_rasters.')
        return

    # ------------------------------------------------------------------
    # Helper: get raster info needed for pygeo write
    # ------------------------------------------------------------------
    def _get_raster_info(path):
        info = pygeo.geoprocessing.get_raster_info(path)
        # pixel_size: (xres, yres) — yres is negative in north-up rasters
        # origin: (x_min, y_max)
        gt = info['geotransform']
        pixel_size = (gt[1], gt[5])
        origin = (gt[0], gt[3])
        return pixel_size, origin, info['projection_wkt'], info['nodata'][0]

    # ------------------------------------------------------------------
    # Helper: write filled raster via pygeoprocessing
    # ------------------------------------------------------------------
    def _write_filled(arr, out_path, ref_path, nodata_val):
        pixel_size, origin, projection_wkt, _ = _get_raster_info(ref_path)
        pygeo.geoprocessing.numpy_array_to_raster(
            arr.astype(np.float32),
            nodata_val,
            pixel_size,
            origin,
            projection_wkt,
            out_path,
            raster_driver_creation_tuple=(
                'GTIFF',
                ('TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                 'BLOCKXSIZE=256', 'BLOCKYSIZE=256')
            )
        )

    # ------------------------------------------------------------------
    # Helper: nearest-neighbor fill for small gaps (ERA5)
    # ------------------------------------------------------------------
    def _nn_fill(arr, nodata_mask):
        if not np.any(nodata_mask):
            return arr
        tmp = arr.astype(np.float64)
        tmp[nodata_mask] = np.nan

        def _local_mean(values):
            centre = values[len(values) // 2]
            if not np.isnan(centre):
                return centre
            valid = values[np.isfinite(values)]
            return float(np.mean(valid)) if len(valid) > 0 else np.nan

        filled = generic_filter(tmp, _local_mean, size=3, mode='nearest')
        arr = arr.copy()
        arr[nodata_mask] = filled[nodata_mask]
        return arr.astype(np.float32)

    # ------------------------------------------------------------------
    # Build land mask from WorldPop 2019
    # pygeo.geoprocessing.raster_to_numpy_array returns (array, nodata)
    # ------------------------------------------------------------------
    worldpop_ref_path = os.path.join(
        p.s3_proj_dir, 'aligned_rasters', 'ppp_2019_1km_Aggregated.tif'
    )
    hb.log(f'Building land mask from: {worldpop_ref_path}')
    wp_ref_arr = pygeo.geoprocessing.raster_to_numpy_array(worldpop_ref_path)
    wp_ref_arr = wp_ref_arr.astype(np.float32)
    wp_ref_ndv = _get_raster_info(worldpop_ref_path)[3]  # nodata value

    land_mask = np.isfinite(wp_ref_arr)
    if wp_ref_ndv is not None:
        atol = abs(wp_ref_ndv) * 1e-5 if wp_ref_ndv != 0 else 1e-10
        land_mask &= ~np.isclose(wp_ref_arr, wp_ref_ndv, rtol=0, atol=atol)
    land_mask &= (wp_ref_arr >= 0)

    hb.log(f'Land mask: {np.sum(land_mask):,} valid pixels '
           f'({100 * np.sum(land_mask) / land_mask.size:.1f}% of global grid)')
    del wp_ref_arr
    gc.collect()

    # ------------------------------------------------------------------
    # 1. avoided_export — fill river/stream NDV with 0 on land
    # ------------------------------------------------------------------
    hb.log('Filling avoided_export rasters...')
    for year in years:
        out_path = os.path.join(p.s3_proj_dir, 'preprocessed_rasters', f'avoided_export_{year}_filled.tif')
        if s3_file_exists(out_path):
            hb.log(f'  avoided_export {year}: already exists, skipping')
            continue

        in_path = os.path.join(p.s3_proj_dir, 'aligned_rasters', f'avoided_export_{year}.tif')
        arr = pygeo.geoprocessing.raster_to_numpy_array(in_path)
        arr = arr.astype(np.float32)
        ndv = _get_raster_info(in_path)[3]

        nodata_mask = ~np.isfinite(arr)
        if ndv is not None:
            atol = abs(ndv) * 1e-5 if ndv != 0 else 1e-10
            nodata_mask |= np.isclose(arr, ndv, rtol=0, atol=atol)

        fill_mask = nodata_mask & land_mask
        arr[fill_mask] = 0.0
        # Anything still nodata (off land) set to NDV for clean output
        arr[nodata_mask & ~land_mask] = ndv if ndv is not None else -9999.0

        out_ndv = ndv if ndv is not None else -9999.0
        _write_filled(arr, out_path, in_path, out_ndv)
        hb.log(f'  avoided_export {year}: filled {np.sum(fill_mask):,} pixels with 0')
        del arr
        gc.collect()

    # ------------------------------------------------------------------
    # 2. WorldPop — fill any remaining NDV on land with 0
    # ------------------------------------------------------------------
    hb.log('Filling WorldPop rasters...')
    for year in years:
        out_path = os.path.join(p.s3_proj_dir, 'preprocessed_rasters', f'ppp_{year}_1km_Aggregated_filled.tif')
        if s3_file_exists(out_path):
            hb.log(f'  WorldPop {year}: already exists in s3, skipping')
            continue

        in_path = os.path.join(p.s3_proj_dir, 'aligned_rasters', f'ppp_{year}_1km_Aggregated.tif')
        arr = pygeo.geoprocessing.raster_to_numpy_array(in_path)
        arr = arr.astype(np.float32)
        ndv = _get_raster_info(in_path)[3]

        nodata_mask = ~np.isfinite(arr)
        if ndv is not None:
            atol = abs(ndv) * 1e-5 if ndv != 0 else 1e-10
            nodata_mask |= np.isclose(arr, ndv, rtol=0, atol=atol)
        nodata_mask |= (arr < 0)

        fill_mask = nodata_mask & land_mask
        arr[fill_mask] = 0.0
        arr[nodata_mask & ~land_mask] = ndv if ndv is not None else -9999.0

        out_ndv = ndv if ndv is not None else -9999.0
        _write_filled(arr, out_path, in_path, out_ndv)
        hb.log(f'  WorldPop {year}: filled {np.sum(fill_mask):,} pixels with 0')
        del arr
        gc.collect()

    # ------------------------------------------------------------------
    # 3. ERA5 — nearest-neighbor fill for any gaps
    # ------------------------------------------------------------------
    hb.log('Filling ERA5 rasters...')
    for year in years:
        out_path = os.path.join(p.s3_proj_dir, 'preprocessed_rasters', f'ERA5_annual_precip_{year}_filled.tif')
        if s3_file_exists(out_path):
            hb.log(f'  ERA5 {year}: already exists in s3, skipping')
            continue

        in_path = os.path.join(p.s3_proj_dir, 'aligned_rasters', f'ERA5_annual_precip_{year}.tif')
        arr = pygeo.geoprocessing.raster_to_numpy_array(in_path)
        arr = arr.astype(np.float32)
        ndv = _get_raster_info(in_path)[3]

        nodata_mask = ~np.isfinite(arr)
        if ndv is not None:
            atol = abs(ndv) * 1e-5 if ndv != 0 else 1e-10
            nodata_mask |= np.isclose(arr, ndv, rtol=0, atol=atol)

        n_missing = int(np.sum(nodata_mask & land_mask))
        if n_missing > 0:
            arr = _nn_fill(arr, nodata_mask)
            hb.log(f'  ERA5 {year}: filled {n_missing:,} land pixels via nearest-neighbor')
        else:
            hb.log(f'  ERA5 {year}: no missing land pixels')

        out_ndv = ndv if ndv is not None else -9999.0
        _write_filled(arr, out_path, in_path, out_ndv)
        del arr
        gc.collect()

    hb.log('preprocess_aligned_rasters complete.')

def tile_zones(p):
    """
    Generate list of tile boundaries for processing
    Saves blocks_list.csv with tile definitions
    This is the ITERATOR that creates the zones
    """
    if p.run_this:
        hb.log('Creating tile list from input raster...')
        
        # Get raster info
        raster_path = os.path.join(p.s3_proj_dir, 'aligned_rasters', 'avoided_export_2020.tif')
        ds = gdal.Open(raster_path)
        n_cols = ds.RasterXSize
        n_rows = ds.RasterYSize
        geotransform = ds.GetGeoTransform()
        ds = None
        
        # Calculate tile dimensions
        p.tile_size = p.processing_resolution  # pixels per tile dimension
        
        # Generate tile boundaries
        blocks_list = []
        for row_offset in range(0, n_rows, p.tile_size):
            for col_offset in range(0, n_cols, p.tile_size):
                # Calculate actual tile size (handle edges)
                actual_n_cols = min(p.tile_size, n_cols - col_offset)
                actual_n_rows = min(p.tile_size, n_rows - row_offset)
                
                # Store as [col_offset, row_offset, n_cols, n_rows]
                blocks_list.append([col_offset, row_offset, actual_n_cols, actual_n_rows])
        
        # Save to CSV
        blocks_list_path = os.path.join(p.cur_dir, 'blocks_list.csv')
        hb.python_object_to_csv(blocks_list, blocks_list_path, '2d_list')
        
        hb.log(f'Created {len(blocks_list)} tiles of max size {p.tile_size}x{p.tile_size}')
        hb.log(f'Blocks list saved to: {blocks_list_path}')
        
        # Setup iterator for next task
        # The cur_dir_parent_dir defines where each tile's directory will be
        # We use just the tile name (row_col) for clarity
        p.iterator_replacements = {
            'tile_col_offset': [block[0] for block in blocks_list],
            'tile_row_offset': [block[1] for block in blocks_list],
            'tile_n_cols': [block[2] for block in blocks_list],
            'tile_n_rows': [block[3] for block in blocks_list],
            # Create tile-specific output directories
            # Each tile gets: generate_tile_zones/row_col/
            'cur_dir_parent_dir': [
                p.cur_dir + '/' + f'{block[1]}_{block[0]}'
                for block in blocks_list
            ]
        }

# def estimate_damage_fn(p):
#     """
#     Estimate damage function via Poisson regression for a single tile
#     Runs separate regression for each year
#     Formula: gfld ~ sdr + era5 + worldpop
#     Saves results with year-specific coefficients
#     """
#     if p.run_this:
#         import statsmodels.api as sm
#         import statsmodels.formula.api as smf

#         # Only run if results do not already exist
#         results_path = os.path.join(p.cur_dir, 'tile_damage_function_results.csv')
#         if os.path.exists(results_path):
#             return
        
#         # Get current tile info from iterator
#         col_offset = p.tile_col_offset
#         row_offset = p.tile_row_offset
#         n_cols = p.tile_n_cols
#         n_rows = p.tile_n_rows
        
#         tile_name = f'{row_offset}_{col_offset}'
#         hb.log(f'Processing tile {tile_name}: offset=({row_offset},{col_offset}), size=({n_rows},{n_cols})')
        
#         # Define years with GFLD data
#         panel_years = list(range(2004, 2018))
        
#         # Store results for all years
#         all_results = []
        
#         # Process each year separately
#         for year in panel_years:
#             hb.log(f'  Processing year {year}...')
            
#             # Load rasters for this year
#             try:
#                 # Load GFLD (dependent variable)
#                 gfld_path = os.path.join(p.s3_proj_dir, 'aligned_rasters', f'gfld_1km_{year}.tif')
#                 ds_gfld = gdal.Open(gfld_path)
#                 if ds_gfld is None:
#                     hb.log(f'    ERROR: Could not open {gfld_path}')
#                     continue
#                 band_gfld = ds_gfld.GetRasterBand(1)
#                 gfld_array = band_gfld.ReadAsArray(col_offset, row_offset, n_cols, n_rows).astype(np.float32)
#                 ndv_gfld = band_gfld.GetNoDataValue()
#                 ds_gfld = None
                
#                 # Load SDR (avoided export)
#                 sdr_path = os.path.join(p.s3_proj_dir, 'aligned_rasters', f'avoided_export_{year}.tif')
#                 ds_sdr = gdal.Open(sdr_path)
#                 if ds_sdr is None:
#                     hb.log(f'    ERROR: Could not open {sdr_path}')
#                     continue
#                 band_sdr = ds_sdr.GetRasterBand(1)
#                 sdr_array = band_sdr.ReadAsArray(col_offset, row_offset, n_cols, n_rows).astype(np.float32)
#                 ndv_sdr = band_sdr.GetNoDataValue()
#                 ds_sdr = None
                
#                 # Load ERA5 (precipitation)
#                 era5_path = os.path.join(p.s3_proj_dir, 'aligned_rasters', f'ERA5_annual_precip_{year}.tif')
#                 ds_era5 = gdal.Open(era5_path)
#                 if ds_era5 is None:
#                     hb.log(f'    ERROR: Could not open {era5_path}')
#                     continue
#                 band_era5 = ds_era5.GetRasterBand(1)
#                 era5_array = band_era5.ReadAsArray(col_offset, row_offset, n_cols, n_rows).astype(np.float32)
#                 ndv_era5 = band_era5.GetNoDataValue()
#                 ds_era5 = None
                
#                 # Load WorldPop (population)
#                 worldpop_path = os.path.join(p.s3_proj_dir, 'aligned_rasters', f'ppp_{year}_1km_Aggregated.tif')
#                 ds_worldpop = gdal.Open(worldpop_path)
#                 if ds_worldpop is None:
#                     hb.log(f'    ERROR: Could not open {worldpop_path}')
#                     continue
#                 band_worldpop = ds_worldpop.GetRasterBand(1)
#                 worldpop_array = band_worldpop.ReadAsArray(col_offset, row_offset, n_cols, n_rows).astype(np.float32)
#                 ndv_worldpop = band_worldpop.GetNoDataValue()
#                 ds_worldpop = None
                
#             except Exception as e:
#                 hb.log(f'    ERROR loading rasters for year {year}: {str(e)}')
#                 continue
            
#             # Create valid mask (all variables must be valid)
#             valid_mask = np.ones((n_rows, n_cols), dtype=bool)
            
#             if ndv_gfld is not None:
#                 valid_mask &= (gfld_array != ndv_gfld)
#             if ndv_sdr is not None:
#                 valid_mask &= (sdr_array != ndv_sdr)
#             if ndv_era5 is not None:
#                 valid_mask &= (era5_array != ndv_era5)
#             if ndv_worldpop is not None:
#                 valid_mask &= (worldpop_array != ndv_worldpop)
            
#             # Check for finite values
#             valid_mask &= (np.isfinite(gfld_array) & 
#                           np.isfinite(sdr_array) & 
#                           np.isfinite(era5_array) & 
#                           np.isfinite(worldpop_array))
            
#             # Additional validation: remove extreme values that could cause numerical issues
#             # Remove values that are too large or negative (where they shouldn't be)
#             valid_mask &= (gfld_array >= 0)  # Fatalities can't be negative
#             valid_mask &= (worldpop_array >= 0)  # Population can't be negative
            
#             # Remove extreme outliers (optional - adjust thresholds as needed)
#             valid_mask &= (sdr_array < 1e10)  # Reasonable upper bound
#             valid_mask &= (era5_array < 1e10)
#             valid_mask &= (worldpop_array < 1e10)
            
#             # Extract valid pixels
#             n_valid = np.sum(valid_mask)
            
#             if n_valid == 0:
#                 hb.log(f'    Year {year}: No valid pixels - skipping')
                
#                 # Add skipped results for this year
#                 for var_name in ['Intercept', 'sdr', 'era5', 'worldpop']:
#                     all_results.append({
#                         'tile_name': tile_name,
#                         'year': year,
#                         'variable': var_name,
#                         'coefficient': np.nan,
#                         'se': np.nan,
#                         't_stat': np.nan,
#                         'p_value': np.nan,
#                         'n_obs': 0,
#                         'pseudo_r2': np.nan,
#                         'deviance': np.nan,
#                         'skipped': True,
#                         'skip_reason': 'no_valid_data'
#                     })
#                 continue
            
#             # Build DataFrame for this year
#             year_data = []
#             for i in range(n_rows):
#                 for j in range(n_cols):
#                     if valid_mask[i, j]:
#                         year_data.append({
#                             'gfld': float(gfld_array[i, j]),
#                             'sdr': float(sdr_array[i, j]),
#                             'era5': float(era5_array[i, j]),
#                             'worldpop': float(worldpop_array[i, j])
#                         })
            
#             year_df = pd.DataFrame(year_data)
#             n_obs = len(year_df)
            
#             # DATA VALIDATION CHECKS
#             skip_reason = None
            
#             # Check 1: Minimum observations
#             min_obs = 30
#             if n_obs < min_obs:
#                 skip_reason = f'insufficient_data_{n_obs}_obs'
            
#             # Check 2: Variance in predictors (avoid zero variance)
#             elif year_df['sdr'].std() < 1e-10:
#                 skip_reason = 'zero_variance_sdr'
#             elif year_df['era5'].std() < 1e-10:
#                 skip_reason = 'zero_variance_era5'
#             elif year_df['worldpop'].std() < 1e-10:
#                 skip_reason = 'zero_variance_worldpop'
            
#             # Check 3: Check if there's any variation in outcome
#             elif year_df['gfld'].sum() == 0:
#                 skip_reason = 'zero_fatalities'
#             elif year_df['gfld'].std() < 1e-10:
#                 skip_reason = 'zero_variance_gfld'
            
#             # Check 4: Check for extreme values in predictors
#             elif year_df['sdr'].max() > 1e8:
#                 skip_reason = 'extreme_values_sdr'
#             elif year_df['era5'].max() > 1e8:
#                 skip_reason = 'extreme_values_era5'
#             elif year_df['worldpop'].max() > 1e8:
#                 skip_reason = 'extreme_values_worldpop'
            
#             if skip_reason:
#                 hb.log(f'    Year {year}: {skip_reason} - skipping regression')
                
#                 for var_name in ['Intercept', 'sdr', 'era5', 'worldpop']:
#                     all_results.append({
#                         'tile_name': tile_name,
#                         'year': year,
#                         'variable': var_name,
#                         'coefficient': np.nan,
#                         'se': np.nan,
#                         't_stat': np.nan,
#                         'p_value': np.nan,
#                         'n_obs': n_obs,
#                         'pseudo_r2': np.nan,
#                         'deviance': np.nan,
#                         'skipped': True,
#                         'skip_reason': skip_reason
#                     })
#                 continue
            
#             hb.log(f'    Year {year}: {n_obs:,} valid observations')
#             hb.log(f'      gfld: sum={year_df["gfld"].sum():.1f}, mean={year_df["gfld"].mean():.4f}, std={year_df["gfld"].std():.4f}')
#             hb.log(f'      sdr: mean={year_df["sdr"].mean():.2f}, std={year_df["sdr"].std():.2f}')
#             hb.log(f'      era5: mean={year_df["era5"].mean():.2f}, std={year_df["era5"].std():.2f}')
#             hb.log(f'      worldpop: mean={year_df["worldpop"].mean():.2f}, std={year_df["worldpop"].std():.2f}')
            
#             # Fit Poisson regression for this year
#             try:
#                 formula = 'gfld ~ sdr + era5 + worldpop'
                
#                 # Use fit_regularized if standard fit fails
#                 try:
#                     model = smf.glm(
#                         formula=formula,
#                         data=year_df,
#                         family=sm.families.Poisson()
#                     ).fit()
                    
#                 except (np.linalg.LinAlgError, ValueError) as e:
#                     # Try with regularization if standard fit fails
#                     hb.log(f'    Year {year}: Standard fit failed, trying with regularization...')
#                     model = smf.glm(
#                         formula=formula,
#                         data=year_df,
#                         family=sm.families.Poisson()
#                     ).fit_regularized(alpha=0.01)
                
#                 # Extract coefficients and statistics
#                 coefficients = model.params
#                 se_coef = model.bse if hasattr(model, 'bse') else pd.Series({k: np.nan for k in coefficients.index})
#                 t_stats = model.tvalues if hasattr(model, 'tvalues') else pd.Series({k: np.nan for k in coefficients.index})
#                 p_values = model.pvalues if hasattr(model, 'pvalues') else pd.Series({k: np.nan for k in coefficients.index})
                
#                 # Calculate pseudo R-squared (McFadden's R²)
#                 null_deviance = model.null_deviance if hasattr(model, 'null_deviance') else np.nan
#                 deviance = model.deviance if hasattr(model, 'deviance') else np.nan
#                 pseudo_r2 = 1 - (deviance / null_deviance) if (null_deviance > 0 and np.isfinite(null_deviance)) else np.nan
                
#                 # Check for NaN coefficients
#                 if any(np.isnan(coefficients)):
#                     raise ValueError("Model produced NaN coefficients")
                
#                 # Store results for this year
#                 for var_name in ['Intercept', 'sdr', 'era5', 'worldpop']:
#                     all_results.append({
#                         'tile_name': tile_name,
#                         'year': year,
#                         'variable': var_name,
#                         'coefficient': coefficients[var_name],
#                         'se': se_coef[var_name] if var_name in se_coef else np.nan,
#                         't_stat': t_stats[var_name] if var_name in t_stats else np.nan,
#                         'p_value': p_values[var_name] if var_name in p_values else np.nan,
#                         'n_obs': n_obs,
#                         'pseudo_r2': pseudo_r2,
#                         'deviance': deviance,
#                         'skipped': False,
#                         'skip_reason': None
#                     })
                
#                 hb.log(f'    Year {year}: Regression complete - Pseudo-R²={pseudo_r2:.4f}')
#                 hb.log(f'      sdr={coefficients["sdr"]:.6f}, era5={coefficients["era5"]:.6f}, worldpop={coefficients["worldpop"]:.6f}')
                
#             except Exception as e:
#                 hb.log(f'    Year {year}: Regression failed - {str(e)[:100]}')
                
#                 for var_name in ['Intercept', 'sdr', 'era5', 'worldpop']:
#                     all_results.append({
#                         'tile_name': tile_name,
#                         'year': year,
#                         'variable': var_name,
#                         'coefficient': np.nan,
#                         'se': np.nan,
#                         't_stat': np.nan,
#                         'p_value': np.nan,
#                         'n_obs': n_obs,
#                         'pseudo_r2': np.nan,
#                         'deviance': np.nan,
#                         'skipped': True,
#                         'skip_reason': f'regression_failed_{str(e)[:50]}'
#                     })
            
#             # Clean up arrays to free memory
#             del gfld_array, sdr_array, era5_array, worldpop_array, year_df
#             import gc
#             gc.collect()
        
#         # Convert all results to DataFrame and save
#         if len(all_results) == 0:
#             hb.log(f'Tile {tile_name}: No results for any year')
            
#             # Create empty results
#             results = pd.DataFrame([{
#                 'tile_name': tile_name,
#                 'year': np.nan,
#                 'variable': 'Intercept',
#                 'coefficient': np.nan,
#                 'se': np.nan,
#                 't_stat': np.nan,
#                 'p_value': np.nan,
#                 'n_obs': 0,
#                 'pseudo_r2': np.nan,
#                 'deviance': np.nan,
#                 'skipped': True,
#                 'skip_reason': 'no_years_processed'
#             }])
#         else:
#             results = pd.DataFrame(all_results)
        
#         # Save results to CSV
#         results.to_csv(results_path, index=False)
        
#         # Summary statistics for this tile
#         successful_years = results[results['skipped'] == False]['year'].unique()
#         if len(successful_years) > 0:
#             hb.log(f'Tile {tile_name} complete: {len(successful_years)} years with successful regressions')
#         else:
#             hb.log(f'Tile {tile_name} complete: No successful regressions')
        
#         hb.log(f'Saved results to: {results_path}')


# def aggregate_damage_fn_coefficients(p):
#     """
#     Aggregate regression coefficients and statistics across years for each tile.
#     Averages coefficient, se, t_stat, p_value, pseudo_r2, deviance, and n_obs for each variable.
#     """
#     if p.run_this:
#         results_path = os.path.join(p.cur_dir, 'tile_damage_function_results_agg.csv')
#         if os.path.exists(results_path):
#             return
        
#         tile_results_path = os.path.join(p.estimate_damage_fn_dir, 'tile_damage_function_results.csv')
#         if not os.path.exists(tile_results_path):
#             hb.log(f'Warning: Tile results not found for aggregation: {tile_results_path}')
#             return

#         tile_results = pd.read_csv(tile_results_path)
#         valid = tile_results[(tile_results['skipped'] == False) & (tile_results['coefficient'].notnull())]
#         if valid.empty:
#             hb.log(f'No valid coefficients to aggregate for {tile_results_path}')
#             return
#         # Aggregate all relevant statistics
#         agg_cols = {
#             'coefficient': 'mean',
#             'se': 'mean',
#             't_stat': 'mean',
#             'p_value': 'mean',
#             'n_obs': 'sum',
#             'pseudo_r2': 'mean',
#             'deviance': 'mean'
#         }
#         avg_df = valid.groupby('variable').agg(agg_cols).reset_index()
#         avg_df.to_csv(results_path, index=False)
#         hb.log(f'Aggregated coefficients and statistics saved to: {results_path}')


# def generate_tile_avoided_mortality(p):
#     """
#     Generate avoided mortality raster for a single tile using averaged regression coefficients.
#     Loads the aggregated coefficients, applies them to the 2019 (or target year) regressors,
#     and saves the predicted avoided mortality raster for the tile.
#     """
#     if not p.run_this:
#         return

#     # Get current tile info from iterator
#     col_offset = p.tile_col_offset
#     row_offset = p.tile_row_offset
#     n_cols = p.tile_n_cols
#     n_rows = p.tile_n_rows
#     tile_name = f'{row_offset}_{col_offset}'
#     hb.log(f'Generating avoided mortality for tile {tile_name}')

#     # Only run if results do not already exist
#     output_path = os.path.join(p.cur_dir, f'avoided_mortality_tile_{tile_name}.npy')
#     if os.path.exists(output_path):
#         return

#     # Load aggregated coefficients
#     agg_coef_path = os.path.join(p.aggregate_damage_fn_coefficients_dir, 'tile_damage_function_results_agg.csv')
#     if not os.path.exists(agg_coef_path):
#         blank_array = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
#         np.save(output_path, blank_array)
#         hb.log(f'No coefficients found for tile {tile_name}, blank raster saved.')
#         return
#     coefs = pd.read_csv(agg_coef_path).set_index('variable')['coefficient']

#     # Load 2019 regressors for this tile
#     # SDR (avoided export)
#     sdr_path = os.path.join(p.s3_proj_dir, 'aligned_rasters', 'avoided_export_2019.tif')
#     ds_sdr = gdal.Open(sdr_path)
#     band_sdr = ds_sdr.GetRasterBand(1)
#     sdr_array = band_sdr.ReadAsArray(col_offset, row_offset, n_cols, n_rows).astype(np.float32)
#     ndv_sdr = band_sdr.GetNoDataValue()
#     ds_sdr = None

#     # ERA5 (precipitation)
#     era5_path = os.path.join(p.s3_proj_dir, 'aligned_rasters', 'ERA5_annual_precip_2019.tif')
#     ds_era5 = gdal.Open(era5_path)
#     band_era5 = ds_era5.GetRasterBand(1)
#     era5_array = band_era5.ReadAsArray(col_offset, row_offset, n_cols, n_rows).astype(np.float32)
#     ndv_era5 = band_era5.GetNoDataValue()
#     ds_era5 = None

#     # WorldPop (population)
#     worldpop_path = os.path.join(p.s3_proj_dir, 'aligned_rasters', 'ppp_2019_1km_Aggregated.tif')
#     ds_worldpop = gdal.Open(worldpop_path)
#     band_worldpop = ds_worldpop.GetRasterBand(1)
#     worldpop_array = band_worldpop.ReadAsArray(col_offset, row_offset, n_cols, n_rows).astype(np.float32)
#     ndv_worldpop = band_worldpop.GetNoDataValue()
#     ds_worldpop = None

#     # Create valid mask (all variables must be valid)
#     valid_mask = np.ones((n_rows, n_cols), dtype=bool)
#     if ndv_sdr is not None:
#         valid_mask &= (sdr_array != ndv_sdr)
#     if ndv_era5 is not None:
#         valid_mask &= (era5_array != ndv_era5)
#     if ndv_worldpop is not None:
#         valid_mask &= (worldpop_array != ndv_worldpop)
#     valid_mask &= np.isfinite(sdr_array) & np.isfinite(era5_array) & np.isfinite(worldpop_array)
#     valid_mask &= (sdr_array < 1e10) & (era5_array < 1e10) & (worldpop_array < 1e10)
#     valid_mask &= (worldpop_array >= 0)

#     # DEBUG
#     print("SDR unique values:", np.unique(sdr_array[~np.isnan(sdr_array)])[:10])
#     print("ERA5 unique values:", np.unique(era5_array[~np.isnan(era5_array)])[:10])
#     print("WorldPop unique values:", np.unique(worldpop_array[~np.isnan(worldpop_array)])[:10])
#     print("SDR nodata:", ndv_sdr, "ERA5 nodata:", ndv_era5, "WorldPop nodata:", ndv_worldpop)
#     print("Valid SDR pixels:", np.sum((sdr_array != ndv_sdr) & np.isfinite(sdr_array)))
#     print("Valid ERA5 pixels:", np.sum((era5_array != ndv_era5) & np.isfinite(era5_array)))
#     print("Valid WorldPop pixels:", np.sum((worldpop_array != ndv_worldpop) & np.isfinite(worldpop_array)))
#     print("Valid pixels after all masks:", np.sum(valid_mask))

#     # Build linear predictor: intercept + sdr*coef + era5*coef + worldpop*coef
#     pred = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
#     intercept = coefs.get('Intercept', 0.0)
#     pred[valid_mask] = intercept
#     if 'sdr' in coefs:
#         pred[valid_mask] += coefs['sdr'] * sdr_array[valid_mask]
#     if 'era5' in coefs:
#         pred[valid_mask] += coefs['era5'] * era5_array[valid_mask]
#     if 'worldpop' in coefs:
#         pred[valid_mask] += coefs['worldpop'] * worldpop_array[valid_mask]

#     # Poisson regression: expected count = exp(linear predictor)
#     avoided_mortality = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
#     avoided_mortality[valid_mask] = np.exp(pred[valid_mask])

#     np.save(output_path, avoided_mortality)
#     hb.log(f'Saved avoided mortality tile to: {output_path}')

#     # Clean up
#     del sdr_array, era5_array, worldpop_array, pred, avoided_mortality
#     import gc
#     gc.collect()

def estimate_damage_fn(p):
    """
    Estimate hurdle model damage function for a single tile.
    
    Part 1 - Logistic regression on full panel:
        P(gfld > 0) ~ avoided_export_z + era5_z + worldpop_z
    
    Part 2 - Gamma GLM (log link) on positive-only subset:
        E[gfld | gfld > 0] ~ avoided_export_z + era5_z + worldpop_z
    
    Predictors are standardized using panel-wide mean/std.
    Scaling params are saved alongside coefficients for use at prediction time.
    """
    if p.run_this:
        import statsmodels.api as sm
        from statsmodels.discrete.discrete_model import Logit
        import warnings
        import gc

        results_path = os.path.join(p.cur_dir, 'tile_damage_function_results.csv')
        if os.path.exists(results_path):
            return

        col_offset = p.tile_col_offset
        row_offset = p.tile_row_offset
        n_cols = p.tile_n_cols
        n_rows = p.tile_n_rows
        tile_name = f'{row_offset}_{col_offset}'
        hb.log(f'Processing tile {tile_name}: offset=({row_offset},{col_offset}), size=({n_rows},{n_cols})')

        panel_years = list(range(2004, 2018))
        all_year_dfs = []

        # ------------------------------------------------------------------
        # Load and stack all years
        # ------------------------------------------------------------------
        for year in panel_years:
            try:
                gfld_path     = os.path.join(p.s3_proj_dir, 'aligned_rasters', f'gfld_1km_{year}.tif')
                sdr_path      = os.path.join(p.s3_proj_dir, 'preprocessed_rasters', f'avoided_export_{year}_filled.tif')
                era5_path     = os.path.join(p.s3_proj_dir, 'preprocessed_rasters', f'ERA5_annual_precip_{year}_filled.tif')
                worldpop_path = os.path.join(p.s3_proj_dir, 'preprocessed_rasters', f'ppp_{year}_1km_Aggregated_filled.tif')

                arrays, ndvs, failed = {}, {}, False
                for name, path in [('gfld', gfld_path), ('sdr', sdr_path),
                                    ('era5', era5_path), ('worldpop', worldpop_path)]:
                    ds = gdal.Open(path)
                    if ds is None:
                        hb.log(f'  Year {year}: Could not open {path}, skipping year')
                        failed = True
                        break
                    band = ds.GetRasterBand(1)
                    arrays[name] = band.ReadAsArray(col_offset, row_offset, n_cols, n_rows).astype(np.float64)
                    ndvs[name] = band.GetNoDataValue()
                    ds = None
                if failed:
                    continue

            except Exception as e:
                hb.log(f'  Year {year}: Error loading rasters - {str(e)[:100]}')
                continue

            # Valid mask — only gfld can have meaningful NDV here since
            # sdr/era5/worldpop have been gap-filled
            valid_mask = np.ones((n_rows, n_cols), dtype=bool)
            for name in ['gfld', 'sdr', 'era5', 'worldpop']:
                ndv = ndvs[name]
                arr = arrays[name]
                if ndv is not None:
                    atol = abs(ndv) * 1e-5 if ndv != 0 else 1e-10
                    valid_mask &= ~np.isclose(arr, ndv, rtol=0, atol=atol)
                valid_mask &= np.isfinite(arr)
            valid_mask &= (arrays['gfld'] >= 0)
            valid_mask &= (arrays['worldpop'] >= 0)

            if np.sum(valid_mask) == 0:
                hb.log(f'  Year {year}: No valid pixels, skipping')
                continue

            year_df = pd.DataFrame({
                'gfld':     arrays['gfld'][valid_mask],
                'sdr':      arrays['sdr'][valid_mask],
                'era5':     arrays['era5'][valid_mask],
                'worldpop': arrays['worldpop'][valid_mask],
                'year':     year
            })
            all_year_dfs.append(year_df)
            del arrays
            gc.collect()

        # ------------------------------------------------------------------
        # Helper to write a skipped result
        # ------------------------------------------------------------------
        def _write_skipped(reason):
            rows = []
            for part in ['logit', 'gamma']:
                for var in ['Intercept', 'sdr_z', 'era5_z', 'worldpop_z']:
                    rows.append({
                        'tile_name': tile_name, 'model_part': part, 'variable': var,
                        'coefficient': np.nan, 'se': np.nan, 't_stat': np.nan,
                        'p_value': np.nan, 'n_obs_total': 0, 'n_obs_positive': 0,
                        'aic_logit': np.nan, 'aic_gamma': np.nan,
                        'skipped': True, 'skip_reason': reason,
                        'sdr_mean': np.nan, 'sdr_std': np.nan,
                        'era5_mean': np.nan, 'era5_std': np.nan,
                        'worldpop_mean': np.nan, 'worldpop_std': np.nan,
                    })
            pd.DataFrame(rows).to_csv(results_path, index=False)
            hb.log(f'Tile {tile_name}: skipped - {reason}')

        if len(all_year_dfs) == 0:
            _write_skipped('no_valid_years')
            return

        panel_df = pd.concat(all_year_dfs, ignore_index=True)
        n_obs_total = len(panel_df)
        n_obs_positive = int((panel_df['gfld'] > 0).sum())
        hb.log(f'Tile {tile_name}: {n_obs_total:,} total obs, '
               f'{n_obs_positive:,} positive ({100*n_obs_positive/n_obs_total:.3f}%)')

        if n_obs_total < 50:
            _write_skipped(f'insufficient_obs_{n_obs_total}')
            return
        if n_obs_positive < 10:
            _write_skipped(f'insufficient_positive_obs_{n_obs_positive}')
            return

        # ------------------------------------------------------------------
        # Standardize predictors, save scaling params
        # ------------------------------------------------------------------
        scaling = {}
        for var in ['sdr', 'era5', 'worldpop']:
            mu = panel_df[var].mean()
            sd = panel_df[var].std()
            if sd < 1e-10:
                _write_skipped(f'zero_variance_{var}')
                return
            panel_df[f'{var}_z'] = (panel_df[var] - mu) / sd
            scaling[f'{var}_mean'] = mu
            scaling[f'{var}_std'] = sd

        exog_vars = ['sdr_z', 'era5_z', 'worldpop_z']
        exog = sm.add_constant(panel_df[exog_vars], has_constant='add')

        # ------------------------------------------------------------------
        # Part 1: Logistic regression — P(gfld > 0)
        # ------------------------------------------------------------------
        binary_y = (panel_df['gfld'] > 0).astype(int)
        logit_result = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                logit_result = Logit(binary_y, exog).fit(
                    method='bfgs', maxiter=200, disp=False
                )
            hb.log(f'  Logit: converged={logit_result.mle_retvals.get("converged","?")}, '
                   f'AIC={logit_result.aic:.2f}')
        except Exception as e:
            _write_skipped(f'logit_failed_{str(e)[:80]}')
            return

        # ------------------------------------------------------------------
        # Part 2: Gamma GLM — E[gfld | gfld > 0]
        # ------------------------------------------------------------------
        positive_df = panel_df[panel_df['gfld'] > 0].copy()
        exog_pos = sm.add_constant(positive_df[exog_vars], has_constant='add')
        gamma_result = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                gamma_result = sm.GLM(
                    positive_df['gfld'],
                    exog_pos,
                    family=sm.families.Gamma(link=sm.families.links.Log())
                ).fit(method='irls', maxiter=200, disp=False)
            hb.log(f'  Gamma: converged={gamma_result.converged}, '
                   f'AIC={gamma_result.aic:.2f}')
        except Exception as e:
            _write_skipped(f'gamma_failed_{str(e)[:80]}')
            return

        # ------------------------------------------------------------------
        # Store results — one row per model_part x variable
        # ------------------------------------------------------------------
        rows = []
        for model_part, result in [('logit', logit_result), ('gamma', gamma_result)]:
            coefficients = result.params
            se_coef  = result.bse    if hasattr(result, 'bse')    else pd.Series(dtype=float)
            t_stats  = result.tvalues if hasattr(result, 'tvalues') else pd.Series(dtype=float)
            p_values = result.pvalues if hasattr(result, 'pvalues') else pd.Series(dtype=float)
            aic_logit = logit_result.aic
            aic_gamma = gamma_result.aic

            for raw_name, friendly in [('const', 'Intercept'),
                                        ('sdr_z',      'sdr_z'),
                                        ('era5_z',     'era5_z'),
                                        ('worldpop_z', 'worldpop_z')]:
                rows.append({
                    'tile_name':      tile_name,
                    'model_part':     model_part,
                    'variable':       friendly,
                    'coefficient':    coefficients.get(raw_name, np.nan),
                    'se':             se_coef.get(raw_name, np.nan),
                    't_stat':         t_stats.get(raw_name, np.nan),
                    'p_value':        p_values.get(raw_name, np.nan),
                    'n_obs_total':    n_obs_total,
                    'n_obs_positive': n_obs_positive,
                    'aic_logit':      aic_logit,
                    'aic_gamma':      aic_gamma,
                    'skipped':        False,
                    'skip_reason':    None,
                    **scaling
                })

        pd.DataFrame(rows).to_csv(results_path, index=False)
        hb.log(f'Saved results to: {results_path}')

def _read_tile(path, col_offset, row_offset, n_cols, n_rows):
    """
    Read a windowed tile from a raster using GDAL.
    Returns (array as float64, nodata_value).
    pygeoprocessing does not support windowed reads, so we use GDAL directly here.
    """
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(f'Could not open raster: {path}')
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray(col_offset, row_offset, n_cols, n_rows).astype(np.float64)
    ndv = band.GetNoDataValue()
    ds = None
    return arr, ndv


def _nodata_mask(arr, ndv):
    """
    Return a boolean mask of nodata pixels (True = nodata).
    Uses np.isclose for float NDV comparison to avoid precision issues.
    """
    mask = ~np.isfinite(arr)
    if ndv is not None:
        atol = abs(ndv) * 1e-5 if ndv != 0 else 1e-10
        mask |= np.isclose(arr, ndv, rtol=0, atol=atol)
    return mask

def aggregate_damage_fn_coefficients(p):
    """
    Aggregate hurdle model coefficients for a single tile.
    Groups by (model_part, variable) and averages statistics.
    Scaling params (tile-level constants) are passed through from the first valid row.
    Writes tile_damage_function_results_agg.csv for use by generate_tile_avoided_mortality.
    Always writes the file, even if skipped, so downstream tasks never crash on missing file.
    """
    if not p.run_this:
        return

    results_path = os.path.join(p.cur_dir, 'tile_damage_function_results_agg.csv')
    if os.path.exists(results_path):
        return

    def _write_skipped(reason):
        hb.log(f'  Writing skipped sentinel: {reason}')
        pd.DataFrame([{'model_part': None, 'variable': None, 'coefficient': np.nan,
                       'se': np.nan, 't_stat': np.nan, 'p_value': np.nan,
                       'n_obs_total': 0, 'n_obs_positive': 0,
                       'aic_logit': np.nan, 'aic_gamma': np.nan,
                       'sdr_mean': np.nan, 'sdr_std': np.nan,
                       'era5_mean': np.nan, 'era5_std': np.nan,
                       'worldpop_mean': np.nan, 'worldpop_std': np.nan,
                       'skipped': True}]).to_csv(results_path, index=False)

    tile_results_path = os.path.join(p.estimate_damage_fn_dir, 'tile_damage_function_results.csv')
    if not os.path.exists(tile_results_path):
        hb.log(f'Warning: Tile results not found: {tile_results_path}')
        _write_skipped('tile_results_not_found')
        return

    tile_results = pd.read_csv(tile_results_path)
    valid = tile_results[
        (tile_results['skipped'] == False) &
        (tile_results['coefficient'].notnull())
    ]

    if valid.empty:
        hb.log(f'No valid coefficients to aggregate at {tile_results_path}')
        _write_skipped('no_valid_coefficients')
        return

    # Group by model_part and variable — one row per (logit/gamma x variable)
    agg_cols = {
        'coefficient':    'mean',
        'se':             'mean',
        't_stat':         'mean',
        'p_value':        'mean',
        'n_obs_total':    'first',
        'n_obs_positive': 'first',
        'aic_logit':      'first',
        'aic_gamma':      'first',
    }
    avg_df = valid.groupby(['model_part', 'variable']).agg(agg_cols).reset_index()

    scaling_cols = ['sdr_mean', 'sdr_std', 'era5_mean', 'era5_std', 'worldpop_mean', 'worldpop_std']
    first_row = valid.iloc[0]
    for col in scaling_cols:
        avg_df[col] = first_row[col] if col in first_row.index else np.nan

    avg_df['skipped'] = False
    avg_df.to_csv(results_path, index=False)

    n_obs = avg_df['n_obs_total'].iloc[0]
    n_pos = avg_df['n_obs_positive'].iloc[0]
    aic_l = avg_df['aic_logit'].iloc[0]
    aic_g = avg_df['aic_gamma'].iloc[0]
    hb.log(f'Aggregated hurdle coefficients saved to: {results_path}')
    hb.log(f'  n_obs={n_obs:,}, n_positive={n_pos:,}, AIC_logit={aic_l:.2f}, AIC_gamma={aic_g:.2f}')
    for part in ['logit', 'gamma']:
        part_df = avg_df[avg_df['model_part'] == part]
        hb.log(f'  [{part}]')
        for _, row in part_df.iterrows():
            hb.log(f'    {row["variable"]:15s}: {row["coefficient"]:10.6f} '
                   f'(SE: {row["se"]:.6f}, p: {row["p_value"]:.4f})')


# def generate_tile_avoided_mortality(p):
#     """
#     Generate avoided mortality raster for a single tile using the hurdle model.

#     Prediction:
#         E[gfld] = P(gfld > 0) * E[gfld | gfld > 0]
#                = logistic(X @ logit_coefs) * exp(X @ gamma_coefs)

#     Loads aggregated hurdle coefficients and scaling params, standardizes 2019
#     predictors using training-time scaling, and saves predicted expected mortality
#     as a float32 .npy array. NaN where any predictor is missing.
#     """
#     hb.log(f'ENTERED generate_tile_avoided_mortality, run_this={p.run_this}')
#     if not p.run_this:
#         return

#     col_offset = p.tile_col_offset
#     row_offset = p.tile_row_offset
#     n_cols     = p.tile_n_cols
#     n_rows     = p.tile_n_rows
#     tile_name  = f'{row_offset}_{col_offset}'
#     hb.log(f'TILE {tile_name}: starting')

#     output_path = os.path.join(p.cur_dir, f'avoided_mortality_tile_{tile_name}.npy')
#     hb.log(f'TILE {tile_name}: output_path={output_path}, exists={os.path.exists(output_path)}')

#     agg_coef_path = os.path.join(p.aggregate_damage_fn_coefficients_dir, 'tile_damage_function_results_agg.csv')
#     hb.log(f'TILE {tile_name}: agg_coef_path={agg_coef_path}, exists={os.path.exists(agg_coef_path)}')

#     if not os.path.exists(agg_coef_path):
#         hb.log(f'TILE {tile_name}: BAIL - no coef file')
#         return

#     coef_df = pd.read_csv(agg_coef_path)
#     coef_df['skipped'] = coef_df['skipped'].map({'True': True, 'False': False})
#     hb.log(f'TILE {tile_name}: skipped.all()={coef_df["skipped"].all()}, skipped values={coef_df["skipped"].tolist()}')

#     if coef_df['skipped'].all():
#         hb.log(f'TILE {tile_name}: BAIL - all skipped')
#         return
    
#     if not p.run_this:
#         return

#     col_offset = p.tile_col_offset
#     row_offset = p.tile_row_offset
#     n_cols     = p.tile_n_cols
#     n_rows     = p.tile_n_rows
#     tile_name  = f'{row_offset}_{col_offset}'

#     output_path = os.path.join(p.cur_dir, f'avoided_mortality_tile_{tile_name}.npy')
#     if os.path.exists(output_path):
#         return

#     hb.log(f'Generating avoided mortality for tile {tile_name}')

#     # ------------------------------------------------------------------
#     # Load aggregated hurdle coefficients
#     # ------------------------------------------------------------------
#     agg_coef_path = os.path.join(
#         p.aggregate_damage_fn_coefficients_dir,
#         'tile_damage_function_results_agg.csv'
#     )
#     if not os.path.exists(agg_coef_path):
#         hb.log(f'  No coefficients found for tile {tile_name}, skipping.')
#         return

#     coef_df = pd.read_csv(agg_coef_path)

#     # CSV round-trip converts booleans to strings — normalize back
#     coef_df['skipped'] = coef_df['skipped'].apply(
#         lambda x: x if isinstance(x, bool) else str(x).strip() == 'True'
#     )

#     if coef_df['skipped'].all():
#         hb.log(f'  All results skipped for tile {tile_name}, skipping prediction.')
#         return

#     coef_df = coef_df[~coef_df['skipped']]

def generate_tile_avoided_mortality(p):
    if not p.run_this:
        return

    # Capture ALL iterator-injected values immediately before any async mutation
    col_offset  = int(p.tile_col_offset)
    row_offset  = int(p.tile_row_offset)
    n_cols      = int(p.tile_n_cols)
    n_rows      = int(p.tile_n_rows)
    cur_dir     = str(p.cur_dir)
    agg_coef_dir = str(p.aggregate_damage_fn_coefficients_dir)
    tile_name   = f'{row_offset}_{col_offset}'

    output_path = os.path.join(cur_dir, f'avoided_mortality_tile_{tile_name}.npy')
    if os.path.exists(output_path):
        return

    agg_coef_path = os.path.join(agg_coef_dir, 'tile_damage_function_results_agg.csv')
    if not os.path.exists(agg_coef_path):
        hb.log(f'  Tile {tile_name}: BAIL - no coef file at {agg_coef_path}')
        return

    coef_df = pd.read_csv(agg_coef_path)
    coef_df['skipped'] = coef_df['skipped'].apply(
        lambda x: x if isinstance(x, bool) else str(x).strip() == 'True'
    )

    if coef_df['skipped'].all():
        hb.log(f'  Tile {tile_name}: BAIL - all skipped')
        return

    coef_df = coef_df[~coef_df['skipped']]

    # Extract per-part coefficient series, indexed by variable name
    def _get_coefs(part):
        return coef_df[coef_df['model_part'] == part].set_index('variable')['coefficient']

    logit_coefs = _get_coefs('logit')
    gamma_coefs = _get_coefs('gamma')

    if logit_coefs.empty or gamma_coefs.empty:
        hb.log(f'  Missing logit or gamma coefficients for tile {tile_name}, skipping.')
        return

    # Scaling params — same on every row, read from first row
    meta = coef_df.iloc[0]
    scaling = {
        'sdr':      (meta['sdr_mean'],      meta['sdr_std']),
        'era5':     (meta['era5_mean'],     meta['era5_std']),
        'worldpop': (meta['worldpop_mean'], meta['worldpop_std']),
    }

    # ------------------------------------------------------------------
    # Load 2019 predictor rasters (windowed tile read)
    # ------------------------------------------------------------------
    sdr_path      = os.path.join(p.preprocess_aligned_rasters_dir, 'avoided_export_2019_filled.tif')
    era5_path     = os.path.join(p.preprocess_aligned_rasters_dir, 'ERA5_annual_precip_2019_filled.tif')
    worldpop_path = os.path.join(p.preprocess_aligned_rasters_dir, 'ppp_2019_1km_Aggregated_filled.tif')

    try:
        sdr_arr,      ndv_sdr      = _read_tile(sdr_path,      col_offset, row_offset, n_cols, n_rows)
        era5_arr,     ndv_era5     = _read_tile(era5_path,     col_offset, row_offset, n_cols, n_rows)
        worldpop_arr, ndv_worldpop = _read_tile(worldpop_path, col_offset, row_offset, n_cols, n_rows)
    except Exception as e:
        hb.log(f'  Tile {tile_name}: Error loading 2019 rasters - {str(e)}')
        return

    # ------------------------------------------------------------------
    # Valid mask — after gap-filling, missing pixels are genuinely absent
    # ------------------------------------------------------------------
    valid_mask = (
        ~_nodata_mask(sdr_arr,      ndv_sdr)      &
        ~_nodata_mask(era5_arr,     ndv_era5)     &
        ~_nodata_mask(worldpop_arr, ndv_worldpop) &
        (worldpop_arr >= 0)
    )
    n_valid = int(np.sum(valid_mask))
    hb.log(f'  Tile {tile_name}: {n_valid:,} valid pixels for prediction')

    if n_valid == 0:
        hb.log(f'  Tile {tile_name}: No valid pixels, skipping.')
        return

    # ------------------------------------------------------------------
    # Standardize using training-time scaling params
    # ------------------------------------------------------------------
    def _standardize(arr, var):
        mu, sd = scaling[var]
        return (arr[valid_mask] - mu) / sd

    sdr_z      = _standardize(sdr_arr,      'sdr')
    era5_z     = _standardize(era5_arr,     'era5')
    worldpop_z = _standardize(worldpop_arr, 'worldpop')

    # Design matrix for valid pixels: [intercept, sdr_z, era5_z, worldpop_z]
    ones = np.ones(n_valid)
    X = np.column_stack([ones, sdr_z, era5_z, worldpop_z])
    var_order = ['Intercept', 'sdr_z', 'era5_z', 'worldpop_z']

    logit_b = np.array([logit_coefs.get(v, 0.0) for v in var_order])
    gamma_b = np.array([gamma_coefs.get(v, 0.0) for v in var_order])

    # ------------------------------------------------------------------
    # Hurdle prediction: E[y] = P(y>0) * E[y | y>0]
    # ------------------------------------------------------------------
    # Part 1: logistic sigmoid -> P(gfld > 0)
    eta_logit = np.clip(X @ logit_b, -50, 50)
    p_nonzero = 1.0 / (1.0 + np.exp(-eta_logit))

    # Part 2: Gamma log link -> E[gfld | gfld > 0]
    eta_gamma = np.clip(X @ gamma_b, -50, 50)
    mu_positive = np.exp(eta_gamma)

    # Combined expected value
    expected_mortality = (p_nonzero * mu_positive).astype(np.float32)

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    out_arr = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    out_arr[valid_mask] = expected_mortality

    hb.log(f'  Tile {tile_name}: predicted min={np.nanmin(out_arr):.6f}, '
           f'max={np.nanmax(out_arr):.6f}, mean={np.nanmean(out_arr):.6f}')

    np.save(output_path, out_arr)
    hb.log(f'  Saved: {output_path}')

    del sdr_arr, era5_arr, worldpop_arr, out_arr
    gc.collect()

def aggregate_damage_results(p):
    """
    Aggregate damage function results from all tiles into single summary.
    """
    if p.run_this:
        hb.log('Aggregating damage function results from all tiles...')

        output_path = os.path.join(p.cur_dir, 'all_tiles_damage_function_results.csv')
        if os.path.exists(output_path):
            hb.log(f'Combined results already exist at {output_path}, skipping aggregation')
            return

        tiles_parent_dir = p.tile_zones_dir
        if not os.path.exists(tiles_parent_dir):
            hb.log(f'Warning: Tiles directory not found: {tiles_parent_dir}')
            return

        # Find all tile results
        all_results = []
        n_missing = 0
        for item in sorted(os.listdir(tiles_parent_dir)):
            if item == 'blocks_list.csv':
                continue
            item_path = os.path.join(tiles_parent_dir, item)
            if os.path.isdir(item_path) and '_' in item:
                results_file = os.path.join(item_path, 'estimate_damage_fn', 'tile_damage_function_results.csv')
                if os.path.exists(results_file):
                    try:
                        df = pd.read_csv(results_file)
                        all_results.append(df)
                    except Exception as e:
                        hb.log(f'Warning: Could not read {results_file}: {e}')
                        n_missing += 1

        hb.log(f'Found {len(all_results)} tile results files ({n_missing} unreadable)')

        if not all_results:
            hb.log('No results found, exiting.')
            return

        combined_results = pd.concat(all_results, ignore_index=True)

        # Normalize column names: support both old (n_obs) and new (n_obs_total) schemas
        if 'n_obs' in combined_results.columns and 'n_obs_total' not in combined_results.columns:
            combined_results = combined_results.rename(columns={'n_obs': 'n_obs_total'})
        # Fill any missing expected columns with safe defaults
        for col, default in [('n_obs_total', 0), ('n_obs_positive', 0), ('skipped', True),
                              ('coefficient', np.nan), ('se', np.nan), ('t_stat', np.nan),
                              ('p_value', np.nan), ('aic_logit', np.nan), ('aic_gamma', np.nan)]:
            if col not in combined_results.columns:
                hb.log(f'Warning: column "{col}" missing from results, filling with {default}')
                combined_results[col] = default

        combined_results.to_csv(output_path, index=False)
        hb.log(f'Saved combined results ({len(combined_results)} rows) to: {output_path}')

        # Filter to valid (non-skipped) rows with observations and valid coefficients
        valid = combined_results[
            (combined_results['skipped'] == False) &
            (combined_results['n_obs_total'] > 0) &
            (combined_results['coefficient'].notna())
        ]

        if valid.empty:
            hb.log('No valid (non-skipped) tile results found. Skipping summary.')
            return

        summary_stats = []
        variables = ['Intercept', 'sdr_z', 'era5_z', 'worldpop_z']
        model_parts = valid['model_part'].unique() if 'model_part' in valid.columns else [None]

        for model_part in sorted(model_parts):
            part_data = valid[valid['model_part'] == model_part] if model_part is not None else valid

            for variable in variables:
                var_data = part_data[part_data['variable'] == variable]
                if var_data.empty:
                    continue

                var_clean = var_data[var_data['coefficient'].notna()]
                weighted_coef = (
                    np.average(var_clean['coefficient'], weights=var_clean['n_obs_total'])
                    if not var_clean.empty else np.nan
                )

                summary_stats.append({
                    'model_part':          model_part,
                    'variable':            variable,
                    'mean_coefficient':    weighted_coef,
                    'median_coefficient':  var_data['coefficient'].median(),
                    'std_coefficient':     var_data['coefficient'].std(),
                    'min_coefficient':     var_data['coefficient'].min(),
                    'max_coefficient':     var_data['coefficient'].max(),
                    'mean_se':             var_data['se'].mean(),
                    'mean_t_stat':         var_data['t_stat'].mean(),
                    'mean_p_value':        var_data['p_value'].mean(),
                    'n_tiles_estimated':   len(var_data),
                    'total_obs_total':     int(var_data['n_obs_total'].sum()),
                    'total_obs_positive':  int(var_data['n_obs_positive'].sum()),
                    'mean_aic_logit':      var_data['aic_logit'].mean(),
                    'mean_aic_gamma':      var_data['aic_gamma'].mean(),
                })

        summary_df = pd.DataFrame(summary_stats)
        summary_path = os.path.join(p.cur_dir, 'damage_function_summary_statistics.csv')
        summary_df.to_csv(summary_path, index=False)

        hb.log('=' * 60)
        hb.log('DAMAGE FUNCTION SUMMARY')
        hb.log('=' * 60)

        for model_part in sorted(summary_df['model_part'].unique()):
            hb.log(f'\nModel part: {model_part}')
            part_summary = summary_df[summary_df['model_part'] == model_part]

            for _, row in part_summary.iterrows():
                sig = ('***' if row['mean_p_value'] < 0.001 else
                       '**'  if row['mean_p_value'] < 0.01  else
                       '*'   if row['mean_p_value'] < 0.05  else '')
                hb.log(
                    f"  {row['variable']:15s}: {row['mean_coefficient']:10.6f} {sig:3s} "
                    f"(SE: {row['mean_se']:8.6f}, t: {row['mean_t_stat']:7.2f}, "
                    f"n_tiles: {int(row['n_tiles_estimated'])}, "
                    f"n_obs: {int(row['total_obs_total']):,})"
                )

        hb.log('=' * 60)
        hb.log(f'Summary saved to: {summary_path}')

# def generate_tile_avoided_mortality(p):
#     """
#     Generate avoided mortality raster for a single tile using ZINB coefficients.
#     Loads aggregated coefficients and scaling params, standardizes 2019 predictors
#     using training-time scaling, applies the count-model linear predictor, and
#     saves predicted expected counts as a .npy array.
#     """
#     if not p.run_this:
#         return

#     col_offset = p.tile_col_offset
#     row_offset = p.tile_row_offset
#     n_cols = p.tile_n_cols
#     n_rows = p.tile_n_rows
#     tile_name = f'{row_offset}_{col_offset}'
#     hb.log(f'Generating avoided mortality for tile {tile_name}')

#     output_path = os.path.join(p.cur_dir, f'avoided_mortality_tile_{tile_name}.npy')
#     if os.path.exists(output_path):
#         return

#     # Load aggregated coefficients (written by aggregate_damage_fn_coefficients)
#     agg_coef_path = os.path.join(p.aggregate_damage_fn_coefficients_dir, 'tile_damage_function_results_agg.csv')
#     if not os.path.exists(agg_coef_path):
#         hb.log(f'No coefficients found for tile {tile_name}, skipping.')
#         return

#     coef_df = pd.read_csv(agg_coef_path)
#     if coef_df['skipped'].all():
#         hb.log(f'All results skipped for tile {tile_name}, skipping prediction.')
#         return

#     coef_df = coef_df[~coef_df['skipped']]
#     coefs = coef_df.set_index('variable')['coefficient']

#     # Scaling params are the same for every row (tile-level), just grab first row
#     meta = coef_df.iloc[0]
#     sdr_mean, sdr_std         = meta['sdr_mean'],      meta['sdr_std']
#     era5_mean, era5_std       = meta['era5_mean'],     meta['era5_std']
#     worldpop_mean, worldpop_std = meta['worldpop_mean'], meta['worldpop_std']

#     # --- Load 2019 predictor rasters ---
#     def _load_band(path):
#         ds = gdal.Open(path)
#         if ds is None:
#             raise FileNotFoundError(f'Could not open {path}')
#         band = ds.GetRasterBand(1)
#         arr = band.ReadAsArray(col_offset, row_offset, n_cols, n_rows).astype(np.float64)
#         ndv = band.GetNoDataValue()
#         ds = None
#         return arr, ndv

#     try:
#         sdr_array,      ndv_sdr      = _load_band(os.path.join(p.s3_proj_dir, 'aligned_rasters', 'avoided_export_2019.tif'))
#         era5_array,     ndv_era5     = _load_band(os.path.join(p.s3_proj_dir, 'aligned_rasters', 'ERA5_annual_precip_2019.tif'))
#         worldpop_array, ndv_worldpop = _load_band(os.path.join(p.s3_proj_dir, 'aligned_rasters', 'ppp_2019_1km_Aggregated.tif'))
#     except Exception as e:
#         hb.log(f'Tile {tile_name}: Error loading 2019 rasters - {str(e)}')
#         return

#     # --- Build valid mask using np.isclose for NDV comparisons ---
#     def _ndv_mask(arr, ndv):
#         if ndv is None:
#             return np.zeros(arr.shape, dtype=bool)
#         atol = abs(ndv) * 1e-5 if ndv != 0 else 1e-10
#         return np.isclose(arr, ndv, rtol=0, atol=atol)

#     valid_mask = (
#         ~_ndv_mask(sdr_array, ndv_sdr) &
#         ~_ndv_mask(era5_array, ndv_era5) &
#         ~_ndv_mask(worldpop_array, ndv_worldpop) &
#         np.isfinite(sdr_array) & np.isfinite(era5_array) & np.isfinite(worldpop_array) &
#         (worldpop_array >= 0) &
#         (sdr_array < 1e10) & (era5_array < 1e10) & (worldpop_array < 1e10)
#     )

#     hb.log(f'  Tile {tile_name}: {np.sum(valid_mask):,} valid pixels for prediction')

#     if np.sum(valid_mask) == 0:
#         hb.log(f'  Tile {tile_name}: No valid pixels, skipping.')
#         return

#     # --- Standardize using training-time scaling params ---
#     sdr_z      = (sdr_array[valid_mask]      - sdr_mean)      / sdr_std
#     era5_z     = (era5_array[valid_mask]      - era5_mean)     / era5_std
#     worldpop_z = (worldpop_array[valid_mask]  - worldpop_mean) / worldpop_std

#     # --- Apply count-model linear predictor: eta = intercept + b1*sdr_z + b2*era5_z + b3*worldpop_z ---
#     # Expected count = exp(eta)  (NB log link)
#     intercept   = coefs.get('Intercept',  0.0)
#     b_sdr       = coefs.get('sdr_z',      0.0)
#     b_era5      = coefs.get('era5_z',     0.0)
#     b_worldpop  = coefs.get('worldpop_z', 0.0)

#     eta = intercept + b_sdr * sdr_z + b_era5 * era5_z + b_worldpop * worldpop_z

#     # Clip eta to avoid overflow in exp (float32 overflows above ~88)
#     eta = np.clip(eta, -50, 50)

#     avoided_mortality = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
#     avoided_mortality[valid_mask] = np.exp(eta).astype(np.float32)

#     hb.log(f'  Tile {tile_name}: predicted values min={np.nanmin(avoided_mortality):.4f}, '
#            f'max={np.nanmax(avoided_mortality):.4f}, mean={np.nanmean(avoided_mortality):.4f}')

#     np.save(output_path, avoided_mortality)
#     hb.log(f'Saved avoided mortality tile to: {output_path}')

#     del sdr_array, era5_array, worldpop_array, avoided_mortality
#     import gc
#     gc.collect()


# def aggregate_damage_fn_coefficients(p):
#     """
#     Aggregate regression coefficients and statistics for a single tile.
#     Since we now fit one ZINB panel model per tile (not per year), this mostly
#     cleans up the results and passes scaling parameters through to the output
#     so generate_tile_avoided_mortality can standardize the 2019 predictors correctly.
#     """
#     if p.run_this:
#         results_path = os.path.join(p.cur_dir, 'tile_damage_function_results_agg.csv')
#         if os.path.exists(results_path):
#             return

#         tile_results_path = os.path.join(p.estimate_damage_fn_dir, 'tile_damage_function_results.csv')
#         if not os.path.exists(tile_results_path):
#             hb.log(f'Warning: Tile results not found: {tile_results_path}')
#             return

#         tile_results = pd.read_csv(tile_results_path)
#         valid = tile_results[
#             (tile_results['skipped'] == False) &
#             (tile_results['coefficient'].notnull())
#         ]

#         if valid.empty:
#             hb.log(f'No valid coefficients to aggregate at {tile_results_path}')
#             return

#         # Aggregate model statistics per variable (mean across variables is
#         # only relevant for coefficient/se/t/p; n_obs and aic are tile-level)
#         agg_cols = {
#             'coefficient':    'mean',
#             'se':             'mean',
#             't_stat':         'mean',
#             'p_value':        'mean',
#             'n_obs_total':    'first',
#             'n_obs_positive': 'first',
#             'aic_logit':      'first',
#             'aic_gamma':      'first',
#         }
#         avg_df = valid.groupby(['model_part', 'variable']).agg(agg_cols).reset_index()

#         # Scaling params are tile-level constants — same on every row.
#         # Pull them from the first valid row and broadcast to all variable rows.
#         scaling_cols = ['sdr_mean', 'sdr_std', 'era5_mean', 'era5_std',
#                         'worldpop_mean', 'worldpop_std']
#         first_row = valid.iloc[0]
#         for col in scaling_cols:
#             if col in first_row.index:
#                 avg_df[col] = first_row[col]
#             else:
#                 avg_df[col] = np.nan
#                 hb.log(f'Warning: scaling column {col} not found in tile results')

#         avg_df['skipped'] = False
#         avg_df.to_csv(results_path, index=False)
#         hb.log(f'Aggregated coefficients saved to: {results_path}')
#         hb.log(f'  Variables: {list(avg_df["variable"])}')


# def aggregate_damage_results(p):
#     """
#     Aggregate damage function results from all tiles into single summary
#     Now includes year-specific coefficients
#     """
#     if p.run_this:
#         hb.log('Aggregating damage function results from all tiles...')

#         output_path = os.path.join(p.cur_dir, 'all_tiles_damage_function_results.csv')
#         if os.path.exists(output_path):
#             hb.log(f'Combined results already exist at {output_path}, skipping aggregation')
#             return
        
#         tiles_parent_dir = p.tile_zones_dir
        
#         if not os.path.exists(tiles_parent_dir):
#             hb.log(f'Warning: Tiles directory not found: {tiles_parent_dir}')
#             return
        
#         # Find all tile results
#         all_results = []
        
#         for item in os.listdir(tiles_parent_dir):
#             item_path = os.path.join(tiles_parent_dir, item)
            
#             if item == 'blocks_list.csv':
#                 continue
                
#             if os.path.isdir(item_path) and '_' in item:
#                 damage_fn_dir = os.path.join(item_path, 'estimate_damage_fn')
                
#                 if os.path.exists(damage_fn_dir):
#                     results_file = os.path.join(damage_fn_dir, 'tile_damage_function_results.csv')
                    
#                     if os.path.exists(results_file):
#                         df = pd.read_csv(results_file)
#                         all_results.append(df)
        
#         hb.log(f'Found {len(all_results)} tile results files')
        
#         if all_results:
#             combined_results = pd.concat(all_results, ignore_index=True)
#             combined_results.to_csv(output_path, index=False)
            
#             # Calculate summary statistics by year and variable
#             valid_tiles = combined_results[
#                 (combined_results['skipped'] == False) & 
#                 (combined_results['n_obs'] > 0) &
#                 (combined_results['coefficient'].notna())
#             ]
            
#             if len(valid_tiles) > 0:
#                 summary_stats = []
                
#                 # Group by year and variable
#                 for year in sorted(valid_tiles['year'].unique()):
#                     year_data = valid_tiles[valid_tiles['year'] == year]
                    
#                     for variable in ['Intercept', 'sdr', 'era5', 'worldpop']:
#                         var_data = year_data[year_data['variable'] == variable]
                        
#                         if len(var_data) > 0:
#                             var_data_clean = var_data[var_data['coefficient'].notna()]
                            
#                             if len(var_data_clean) > 0:
#                                 weighted_coef = np.average(var_data_clean['coefficient'], 
#                                                            weights=var_data_clean['n_obs'])
#                             else:
#                                 weighted_coef = np.nan
                            
#                             summary_stats.append({
#                                 'year': int(year),
#                                 'variable': variable,
#                                 'mean_coefficient': weighted_coef,
#                                 'median_coefficient': var_data['coefficient'].median(),
#                                 'std_coefficient': var_data['coefficient'].std(),
#                                 'min_coefficient': var_data['coefficient'].min(),
#                                 'max_coefficient': var_data['coefficient'].max(),
#                                 'mean_se': var_data['se'].mean(),
#                                 'mean_t_stat': var_data['t_stat'].mean(),
#                                 'mean_p_value': var_data['p_value'].mean(),
#                                 'n_tiles_estimated': len(var_data),
#                                 'total_obs': var_data['n_obs'].sum()
#                             })
                
#                 summary_df = pd.DataFrame(summary_stats)
#                 summary_path = os.path.join(p.cur_dir, 'damage_function_summary_statistics.csv')
#                 summary_df.to_csv(summary_path, index=False)
                
#                 hb.log('='*60)
#                 hb.log('DAMAGE FUNCTION SUMMARY (by year)')
#                 hb.log('='*60)
                
#                 # Print summary for each year
#                 for year in sorted(summary_df['year'].unique()):
#                     year_summary = summary_df[summary_df['year'] == year]
#                     total_obs = year_summary['total_obs'].iloc[0]
                    
#                     hb.log(f'\nYear {int(year)}:')
#                     hb.log(f'  Total observations: {int(total_obs):,}')
                    
#                     for _, row in year_summary.iterrows():
#                         sig = '***' if row['mean_p_value'] < 0.001 else '**' if row['mean_p_value'] < 0.01 else '*' if row['mean_p_value'] < 0.05 else ''
#                         hb.log(f'    {row["variable"]:15s}: {row["mean_coefficient"]:10.6f} {sig:3s} '
#                                f'(SE: {row["mean_se"]:8.6f}, t: {row["mean_t_stat"]:7.2f})')
                
#                 hb.log('='*60)
#                 hb.log(f'\nDetailed results saved to: {summary_path}')


def stitch_avoided_mortality_raster(p):
    """
    Stitch together all tile avoided mortality .npy arrays into a global array,
    then save as GeoTIFF using pygeoprocessing.
    """
    if not p.run_this:
        return

    tiff_output_path = os.path.join(p.cur_dir, 'avoided_mortality_global.tif')
    if os.path.exists(tiff_output_path):
        return

    hb.log('Stitching avoided mortality tiles into global array...')
    blocks_list_path = os.path.join(p.tile_zones_dir, 'blocks_list.csv')
    blocks_df = pd.read_csv(blocks_list_path, header=None)
    blocks_df.columns = ['col_offset', 'row_offset', 'n_cols', 'n_rows']
    max_row = int((blocks_df['row_offset'] + blocks_df['n_rows']).max())
    max_col = int((blocks_df['col_offset'] + blocks_df['n_cols']).max())
    global_array = np.full((max_row, max_col), np.nan, dtype=np.float32)
    tiles_found = 0
    tiles_missing = 0
    for idx, row in blocks_df.iterrows():
        col_offset = int(row['col_offset'])
        row_offset = int(row['row_offset'])
        n_cols = int(row['n_cols'])
        n_rows = int(row['n_rows'])
        tile_name = f'{row_offset}_{col_offset}'
        tile_dir = os.path.join(
            p.tile_zones_dir,
            tile_name,
            'generate_tile_avoided_mortality'
        )
        tile_path = os.path.join(tile_dir, f'avoided_mortality_tile_{tile_name}.npy')
        if os.path.exists(tile_path):
            tile_array = np.load(tile_path)
            global_array[row_offset:row_offset+n_rows, col_offset:col_offset+n_cols] = tile_array
            tiles_found += 1
        else:
            tiles_missing += 1
            # hb.log(f'Warning: Missing tile {tile_name} at {tile_path}')
    hb.log(f'Found {tiles_found} tiles to stitch, {tiles_missing} tiles missing')

    # Save as GeoTIFF using pygeoprocessing
    # Use one of the input rasters for geotransform/projection
    ref_raster_path = os.path.join(p.s3_proj_dir, 'aligned_rasters', 'avoided_export_2019.tif')
    ds = gdal.Open(ref_raster_path)
    geotransform = ds.GetGeoTransform()
    projection_wkt = ds.GetProjection()
    ds = None
    pixel_size = (geotransform[1], abs(geotransform[5]))
    origin = (geotransform[0], geotransform[3])
    target_nodata = np.nan

    pygeo.geoprocessing.numpy_array_to_raster(
        global_array,
        target_nodata,
        pixel_size,
        origin,
        projection_wkt,
        tiff_output_path,
        raster_driver_creation_tuple=(
            'GTIFF',
            ('TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256')
        )
    )
    hb.log(f'Stitched global avoided mortality GeoTIFF saved to: {tiff_output_path}')

    # Calculate global statistics
    valid_vals = global_array[np.isfinite(global_array)]
    if len(valid_vals) > 0:
        hb.log('='*60)
        hb.log('GLOBAL AVOIDED MORTALITY ARRAY SUMMARY')
        hb.log('='*60)
        hb.log(f'Output array: {tiff_output_path}')
        hb.log(f'Valid pixels: {len(valid_vals):,}')
        hb.log(f'Mean avoided mortality: {np.mean(valid_vals):.6f}')
        hb.log(f'Median avoided mortality: {np.median(valid_vals):.6f}')
        hb.log(f'Std deviation: {np.std(valid_vals):.6f}')
        hb.log(f'Min avoided mortality: {np.min(valid_vals):.6f}')
        hb.log(f'Max avoided mortality: {np.max(valid_vals):.6f}')
        hb.log(f'5th percentile: {np.percentile(valid_vals, 5):.6f}')
        hb.log(f'95th percentile: {np.percentile(valid_vals, 95):.6f}')
        hb.log('='*60)
    else:
        hb.log('Warning: No valid avoided mortality in global array!')



def compute_value(p, target_year=2019):
    """
    NOT INTEGRATED
    """
    # VSL values by year (million USD)
    # https://www.transportation.gov/office-policy/transportation-policy/revised-departmental-guidance-on-valuation-of-a-statistical-life-in-economic-analysis
    VSL_BY_YEAR = {
        2024: 13.7e6,
        2023: 13.2e6,
        2022: 12.5e6,
        2021: 11.8e6,
        2020: 11.6e6,
        2019: 10.9e6,
        2018: 10.5e6,
        2017: 10.2e6,
        2016: 9.9e6,
        2015: 9.6e6,
        2014: 9.4e6,
        2013: 9.2e6,
        2012: 9.1e6,
    }

    if target_year in VSL_BY_YEAR:
        vsl_usa = VSL_BY_YEAR[target_year]
    else:
        raise ValueError(f"VSL not defined for year {target_year}")

    # Load avoided mortality results
    results_path = os.path.join(p.compute_avoided_mortality_dir, f'avoided_mortality_results_{target_year}.csv')
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Avoided mortality results not found: {results_path}. Run compute_avoided_mortality task first.")
    results = pd.read_csv(results_path)

    # Load GDP data
    # TODO: upload to s3 or define local dir
    gdp_path = os.path.join('.', 'data', 'worldbank_gdp_per_capita.csv')
    gdp_df = pd.read_csv(gdp_path)

    # Load crosswalk between GAUL country codes and iso3
    crosswalk_path = os.path.join('.', 'data', 'GAUL_L0_2024-2014.csv')
    crosswalk_df = pd.read_csv(crosswalk_path, encoding='latin1')
    crosswalk_df = crosswalk_df[crosswalk_df['GAUL_2014'].notna() & crosswalk_df['iso3_code'].notna()]

    # Merge crosswalk to GDP data
    # crosswalk_df = crosswalk_df[['GAUL_2014', 'iso3_code']]
    gdp_merged_df = gdp_df.merge(crosswalk_df, left_on='Country Code', right_on='iso3_code', how='inner')
    print("Unique GAUL_2014 in gdp_merged_df:", gdp_merged_df['GAUL_2014'].unique())

    # Merge GDP data to results
    results = results[~(results['ADM0_NAME'].isna() & results['ADM0_CODE'].isna())]
    results['ADM0_CODE'] = results['ADM0_CODE'].astype(int)
    gdp_merged_df = gdp_merged_df[gdp_merged_df['GAUL_2014'].apply(lambda x: str(x).strip().isdigit())]
    gdp_merged_df['GAUL_2014'] = gdp_merged_df['GAUL_2014'].astype(int)
    print("Unique ADM0_CODE in results:", results['ADM0_CODE'].unique())
    print("Unique GAUL_2014 in gdp_merged_df:", gdp_merged_df['GAUL_2014'].unique())
    results_gdp = results.merge(gdp_merged_df, left_on='ADM0_CODE', right_on='GAUL_2014', how='left')
    # After merge: print rows where GDP data did not merge
    # unmatched_countries_from_results = set(results['ADM0_NAME']) - set(gdp_merged_df['24_admnm'])
    # print("Countries in results not found in GAUL data:", unmatched_countries_from_results)

    # Calculate GDP-adjusted VSL for each region
    gdp_col = f'{target_year} [YR{target_year}]'
    results_gdp[gdp_col] = pd.to_numeric(results_gdp[gdp_col], errors='coerce')
    gdp_per_capita_usa = pd.to_numeric(gdp_df[gdp_df['Country Code'] == 'USA'][gdp_col].values[0], errors='coerce')
    results_gdp['vsl_adjusted'] = vsl_usa * (results_gdp[gdp_col] / gdp_per_capita_usa)

    # Calculate dollar value of avoided mortality
    results_gdp['avoided_mortality_value_usd'] = results_gdp['avoided_mortality'] * results_gdp['vsl_adjusted']
    
    # Save results
    value_results_path = os.path.join(p.cur_dir, f'gdp_adjusted_value_{target_year}.csv')
    results_gdp.to_csv(value_results_path, index=False)

    # Aggregate to ee_r264
    ee_path = os.path.join('.', 'data', 'ee_r264_correspondence.gpkg')
    ee_df = gpd.read_file(ee_path, ignore_geometry=True)
    ee_df = ee_df[['ee_r264_id', 'ee_r264_label', 'ee_r264_name', 'iso3']]
    results_ee = results_gdp.merge(ee_df, left_on='iso3_code', right_on='iso3', how='left')
    print("Countries in results not matched to ee_r264:", set(results_gdp['iso3_code']) - set(ee_df['iso3']))
    results_ee_agg = results_ee.groupby('ee_r264_id').agg({
        'avoided_mortality': 'sum',
        'avoided_mortality_value_usd': 'sum',
        'ee_r264_label': 'first',
        'ee_r264_name': 'first',
        'iso3': 'first'
    }).reset_index()
    results_path = os.path.join(p.cur_dir, f'gdp_adjusted_value_{target_year}_ee.csv')
    results_ee_agg.to_csv(results_path, index=False)
