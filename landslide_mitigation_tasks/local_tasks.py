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
        # p.iterator_replacements = {
        #     'tile_col_offset': [block[0] for block in blocks_list],
        #     'tile_row_offset': [block[1] for block in blocks_list],
        #     'tile_n_cols': [block[2] for block in blocks_list],
        #     'tile_n_rows': [block[3] for block in blocks_list],
        #     # Create tile-specific output directories
        #     # Each tile gets: generate_tile_zones/row_col/
        #     'cur_dir_parent_dir': [
        #         p.cur_dir + '/' + f'{block[1]}_{block[0]}'
        #         for block in blocks_list
        #     ]
        # }
        # Store the TILE BASE directory that all children should branch from
        p.iterator_replacements = {
            'tile_col_offset': [block[0] for block in blocks_list],
            'tile_row_offset': [block[1] for block in blocks_list],
            'tile_n_cols': [block[2] for block in blocks_list],
            'tile_n_rows': [block[3] for block in blocks_list],
            # This sets the base tile directory for ALL child tasks
            'cur_dir_parent_dir': [
                os.path.join(p.cur_dir, f'{block[1]}_{block[0]}')
                for block in blocks_list
            ]
        }

# def tile_regression_and_prediction(p):
#     """
#     Combined task that runs all three steps sequentially for a single tile:
#     1. Estimate damage function (regression)
#     2. Aggregate coefficients 
#     3. Generate avoided mortality predictions
    
#     This avoids hazelbean's iterator chaining issues by keeping everything
#     in one task context.
#     """
#     if not p.run_this:
#         return
    
#     # Capture tile parameters from iterator
#     col_offset = int(p.tile_col_offset)
#     row_offset = int(p.tile_row_offset)
#     n_cols = int(p.tile_n_cols)
#     n_rows = int(p.tile_n_rows)
#     tile_name = f'{row_offset}_{col_offset}'
    
#     # Use p.cur_dir as base for all outputs
#     tile_dir = p.cur_dir
    
#     hb.log(f'Processing tile {tile_name}: offset=({row_offset},{col_offset}), size=({n_rows},{n_cols})')
    
#     # ================================================================
#     # STEP 1: ESTIMATE DAMAGE FUNCTION
#     # ================================================================
#     results_path = os.path.join(tile_dir, 'tile_damage_function_results.csv')
    
#     if not os.path.exists(results_path):
#         import statsmodels.api as sm
#         from statsmodels.discrete.discrete_model import Logit
#         import warnings
        
#         panel_years = list(range(2004, 2018))
#         all_year_dfs = []
        
#         # Load and stack all years
#         for year in panel_years:
#             try:
#                 gfld_path = os.path.join(p.s3_proj_dir, 'aligned_rasters', f'gfld_1km_{year}.tif')
#                 sdr_path = os.path.join(p.s3_proj_dir, 'preprocessed_rasters', f'avoided_export_{year}_filled.tif')
#                 era5_path = os.path.join(p.s3_proj_dir, 'preprocessed_rasters', f'ERA5_annual_precip_{year}_filled.tif')
#                 worldpop_path = os.path.join(p.s3_proj_dir, 'preprocessed_rasters', f'ppp_{year}_1km_Aggregated_filled.tif')
                
#                 arrays, ndvs, failed = {}, {}, False
#                 for name, path in [('gfld', gfld_path), ('sdr', sdr_path),
#                                    ('era5', era5_path), ('worldpop', worldpop_path)]:
#                     ds = gdal.Open(path)
#                     if ds is None:
#                         failed = True
#                         break
#                     band = ds.GetRasterBand(1)
#                     arrays[name] = band.ReadAsArray(col_offset, row_offset, n_cols, n_rows).astype(np.float64)
#                     ndvs[name] = band.GetNoDataValue()
#                     ds = None
#                 if failed:
#                     continue
                    
#             except Exception as e:
#                 hb.log(f'  Year {year}: Error loading rasters - {str(e)[:100]}')
#                 continue
            
#             # Valid mask
#             valid_mask = np.ones((n_rows, n_cols), dtype=bool)
#             for name in ['gfld', 'sdr', 'era5', 'worldpop']:
#                 ndv = ndvs[name]
#                 arr = arrays[name]
#                 if ndv is not None:
#                     atol = abs(ndv) * 1e-5 if ndv != 0 else 1e-10
#                     valid_mask &= ~np.isclose(arr, ndv, rtol=0, atol=atol)
#                 valid_mask &= np.isfinite(arr)
#             valid_mask &= (arrays['gfld'] >= 0)
#             valid_mask &= (arrays['worldpop'] >= 0)
            
#             if np.sum(valid_mask) == 0:
#                 continue
            
#             year_df = pd.DataFrame({
#                 'gfld': arrays['gfld'][valid_mask],
#                 'sdr': arrays['sdr'][valid_mask],
#                 'era5': arrays['era5'][valid_mask],
#                 'worldpop': arrays['worldpop'][valid_mask],
#                 'year': year
#             })
#             all_year_dfs.append(year_df)
#             del arrays
#             gc.collect()

def tile_regression_and_prediction(p):
    """Combined task for regression and prediction"""
    if not p.run_this:
        return
    
    col_offset = int(p.tile_col_offset)
    row_offset = int(p.tile_row_offset)
    n_cols = int(p.tile_n_cols)
    n_rows = int(p.tile_n_rows)
    tile_name = f'{row_offset}_{col_offset}'
    tile_dir = p.cur_dir
    
    hb.log(f'Processing tile {tile_name}: offset=({row_offset},{col_offset}), size=({n_rows},{n_cols})')
    
    results_path = os.path.join(tile_dir, 'tile_damage_function_results.csv')
    
    if not os.path.exists(results_path):
        import statsmodels.api as sm
        from statsmodels.discrete.discrete_model import Logit
        import warnings
        
        panel_years = list(range(2004, 2018))
        all_year_dfs = []
        
        # DIAGNOSTIC: Check if we can access the data directories
        hb.log(f'  Checking data paths:')
        hb.log(f'    s3_proj_dir: {p.s3_proj_dir}')
        hb.log(f'    preprocess_dir: {getattr(p, "preprocess_aligned_rasters_dir", "NOT SET")}')
        
        # Load and stack all years
        for year in panel_years:
            try:
                gfld_path = os.path.join(p.s3_proj_dir, 'aligned_rasters', f'gfld_1km_{year}.tif')
                sdr_path = os.path.join(p.s3_proj_dir, 'preprocessed_rasters', f'avoided_export_{year}_filled.tif')
                era5_path = os.path.join(p.s3_proj_dir, 'preprocessed_rasters', f'ERA5_annual_precip_{year}_filled.tif')
                worldpop_path = os.path.join(p.s3_proj_dir, 'preprocessed_rasters', f'ppp_{year}_1km_Aggregated_filled.tif')
                
                # DIAGNOSTIC: Check first year in detail
                if year == 2004:
                    hb.log(f'  Year {year} paths:')
                    hb.log(f'    gfld: {gfld_path}')
                    hb.log(f'    gfld exists: {os.path.exists(gfld_path) if not gfld_path.startswith("/vsis3") else "S3 path"}')
                
                arrays, ndvs, failed = {}, {}, False
                for name, path in [('gfld', gfld_path), ('sdr', sdr_path),
                                   ('era5', era5_path), ('worldpop', worldpop_path)]:
                    ds = gdal.Open(path)
                    if ds is None:
                        hb.log(f'  Year {year}: Could not open {name} at {path}')
                        failed = True
                        break
                    band = ds.GetRasterBand(1)
                    arrays[name] = band.ReadAsArray(col_offset, row_offset, n_cols, n_rows).astype(np.float64)
                    ndvs[name] = band.GetNoDataValue()
                    ds = None
                    
                    # DIAGNOSTIC: Check first year's data
                    if year == 2004:
                        hb.log(f'    {name}: shape={arrays[name].shape}, '
                               f'min={np.nanmin(arrays[name]):.3f}, '
                               f'max={np.nanmax(arrays[name]):.3f}, '
                               f'n_finite={np.sum(np.isfinite(arrays[name]))}')
                
                if failed:
                    continue
                    
            except Exception as e:
                hb.log(f'  Year {year}: Exception loading rasters - {str(e)}')
                continue
            
            # Valid mask
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
            
            n_valid_this_year = np.sum(valid_mask)
            hb.log(f'  Year {year}: {n_valid_this_year} valid pixels')
            
            if n_valid_this_year == 0:
                continue
            
            year_df = pd.DataFrame({
                'gfld': arrays['gfld'][valid_mask],
                'sdr': arrays['sdr'][valid_mask],
                'era5': arrays['era5'][valid_mask],
                'worldpop': arrays['worldpop'][valid_mask],
                'year': year
            })
            all_year_dfs.append(year_df)
            del arrays
            gc.collect()
        
        hb.log(f'  Total years with valid data: {len(all_year_dfs)}')
        
        ########################################

        # Helper to write skipped result
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
            return  # Exit early
        
        if len(all_year_dfs) == 0:
            _write_skipped('no_valid_years')
            return  # Exit entire function
        
        panel_df = pd.concat(all_year_dfs, ignore_index=True)
        n_obs_total = len(panel_df)
        n_obs_positive = int((panel_df['gfld'] > 0).sum())
        
        if n_obs_total < 50:
            _write_skipped(f'insufficient_obs_{n_obs_total}')
            return
        if n_obs_positive < 10:
            _write_skipped(f'insufficient_positive_obs_{n_obs_positive}')
            return
        
        # Standardize predictors
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
        
        # Logistic regression
        binary_y = (panel_df['gfld'] > 0).astype(int)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                logit_result = Logit(binary_y, exog).fit(method='bfgs', maxiter=200, disp=False)
        except Exception as e:
            _write_skipped(f'logit_failed_{str(e)[:80]}')
            return
        
        # Gamma GLM
        positive_df = panel_df[panel_df['gfld'] > 0].copy()
        exog_pos = sm.add_constant(positive_df[exog_vars], has_constant='add')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                gamma_result = sm.GLM(
                    positive_df['gfld'],
                    exog_pos,
                    family=sm.families.Gamma(link=sm.families.links.Log())
                ).fit(method='irls', maxiter=200, disp=False)
        except Exception as e:
            _write_skipped(f'gamma_failed_{str(e)[:80]}')
            return
        
        # Store results
        rows = []
        for model_part, result in [('logit', logit_result), ('gamma', gamma_result)]:
            coefficients = result.params
            se_coef = result.bse if hasattr(result, 'bse') else pd.Series(dtype=float)
            t_stats = result.tvalues if hasattr(result, 'tvalues') else pd.Series(dtype=float)
            p_values = result.pvalues if hasattr(result, 'pvalues') else pd.Series(dtype=float)
            
            for raw_name, friendly in [('const', 'Intercept'),
                                       ('sdr_z', 'sdr_z'),
                                       ('era5_z', 'era5_z'),
                                       ('worldpop_z', 'worldpop_z')]:
                rows.append({
                    'tile_name': tile_name,
                    'model_part': model_part,
                    'variable': friendly,
                    'coefficient': coefficients.get(raw_name, np.nan),
                    'se': se_coef.get(raw_name, np.nan),
                    't_stat': t_stats.get(raw_name, np.nan),
                    'p_value': p_values.get(raw_name, np.nan),
                    'n_obs_total': n_obs_total,
                    'n_obs_positive': n_obs_positive,
                    'aic_logit': logit_result.aic,
                    'aic_gamma': gamma_result.aic,
                    'skipped': False,
                    'skip_reason': None,
                    **scaling
                })
        
        pd.DataFrame(rows).to_csv(results_path, index=False)
        hb.log(f'Step 1 complete: Saved regression results')
    
    # ================================================================
    # STEP 2: CHECK IF RESULTS ARE VALID (no separate aggregation needed)
    # ================================================================
    coef_df = pd.read_csv(results_path)
    coef_df['skipped'] = coef_df['skipped'].apply(
        lambda x: x if isinstance(x, bool) else str(x).strip() == 'True'
    )
    
    if coef_df['skipped'].all():
        hb.log(f'Tile {tile_name}: Skipped due to insufficient data, no prediction generated')
        return
    
    coef_df = coef_df[~coef_df['skipped']]
    
    # ================================================================
    # STEP 3: GENERATE AVOIDED MORTALITY PREDICTIONS
    # ================================================================
    output_path = os.path.join(tile_dir, f'avoided_mortality_tile_{tile_name}.npy')
    
    if os.path.exists(output_path):
        hb.log(f'Tile {tile_name}: Prediction already exists, skipping')
        return
    
    # Extract coefficients
    def _get_coefs(part):
        return coef_df[coef_df['model_part'] == part].set_index('variable')['coefficient']
    
    logit_coefs = _get_coefs('logit')
    gamma_coefs = _get_coefs('gamma')
    
    if logit_coefs.empty or gamma_coefs.empty:
        hb.log(f'Tile {tile_name}: Missing coefficients, skipping prediction')
        return
    
    # Get scaling params
    meta = coef_df.iloc[0]
    scaling = {
        'sdr': (meta['sdr_mean'], meta['sdr_std']),
        'era5': (meta['era5_mean'], meta['era5_std']),
        'worldpop': (meta['worldpop_mean'], meta['worldpop_std']),
    }
    
    # Load 2019 predictor rasters
    sdr_path = os.path.join(p.s3_proj_dir, 'preprocessed_rasters', 'avoided_export_2019_filled.tif')
    era5_path = os.path.join(p.s3_proj_dir, 'preprocessed_rasters', 'ERA5_annual_precip_2019_filled.tif')
    worldpop_path = os.path.join(p.s3_proj_dir, 'preprocessed_rasters', 'ppp_2019_1km_Aggregated_filled.tif')
    
    try:
        sdr_arr, ndv_sdr = _read_tile(sdr_path, col_offset, row_offset, n_cols, n_rows)
        era5_arr, ndv_era5 = _read_tile(era5_path, col_offset, row_offset, n_cols, n_rows)
        worldpop_arr, ndv_worldpop = _read_tile(worldpop_path, col_offset, row_offset, n_cols, n_rows)
    except Exception as e:
        hb.log(f'Tile {tile_name}: Error loading 2019 rasters - {str(e)}')
        return
    
    # Valid mask
    valid_mask = (
        ~_nodata_mask(sdr_arr, ndv_sdr) &
        ~_nodata_mask(era5_arr, ndv_era5) &
        ~_nodata_mask(worldpop_arr, ndv_worldpop) &
        (worldpop_arr >= 0)
    )
    n_valid = int(np.sum(valid_mask))
    
    if n_valid == 0:
        hb.log(f'Tile {tile_name}: No valid pixels for prediction')
        return
    
    # Standardize
    def _standardize(arr, var):
        mu, sd = scaling[var]
        return (arr[valid_mask] - mu) / sd
    
    sdr_z = _standardize(sdr_arr, 'sdr')
    era5_z = _standardize(era5_arr, 'era5')
    worldpop_z = _standardize(worldpop_arr, 'worldpop')
    
    # Design matrix
    ones = np.ones(n_valid)
    X = np.column_stack([ones, sdr_z, era5_z, worldpop_z])
    var_order = ['Intercept', 'sdr_z', 'era5_z', 'worldpop_z']
    
    logit_b = np.array([logit_coefs.get(v, 0.0) for v in var_order])
    gamma_b = np.array([gamma_coefs.get(v, 0.0) for v in var_order])
    
    # Hurdle prediction
    eta_logit = np.clip(X @ logit_b, -50, 50)
    p_nonzero = 1.0 / (1.0 + np.exp(-eta_logit))
    
    eta_gamma = np.clip(X @ gamma_b, -50, 50)
    mu_positive = np.exp(eta_gamma)
    
    expected_mortality = (p_nonzero * mu_positive).astype(np.float32)
    
    # Write output
    out_arr = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    out_arr[valid_mask] = expected_mortality
    
    np.save(output_path, out_arr)
    hb.log(f'Step 3 complete: Saved prediction (min={np.nanmin(out_arr):.6f}, '
           f'max={np.nanmax(out_arr):.6f}, mean={np.nanmean(out_arr):.6f})')
    
    del sdr_arr, era5_arr, worldpop_arr, out_arr
    gc.collect()

# def estimate_damage_fn(p):
#     """
#     Estimate hurdle model damage function for a single tile.
    
#     Part 1 - Logistic regression on full panel:
#         P(gfld > 0) ~ avoided_export_z + era5_z + worldpop_z
    
#     Part 2 - Gamma GLM (log link) on positive-only subset:
#         E[gfld | gfld > 0] ~ avoided_export_z + era5_z + worldpop_z
    
#     Predictors are standardized using panel-wide mean/std.
#     Scaling params are saved alongside coefficients for use at prediction time.
#     """
#     if p.run_this:
#         import statsmodels.api as sm
#         from statsmodels.discrete.discrete_model import Logit
#         import warnings
#         import gc

#         results_path = os.path.join(p.cur_dir, 'tile_damage_function_results.csv')
#         if os.path.exists(results_path):
#             return

#         col_offset = p.tile_col_offset
#         row_offset = p.tile_row_offset
#         n_cols = p.tile_n_cols
#         n_rows = p.tile_n_rows
#         tile_name = f'{row_offset}_{col_offset}'
#         hb.log(f'Processing tile {tile_name}: offset=({row_offset},{col_offset}), size=({n_rows},{n_cols})')

#         panel_years = list(range(2004, 2018))
#         all_year_dfs = []

#         # ------------------------------------------------------------------
#         # Load and stack all years
#         # ------------------------------------------------------------------
#         for year in panel_years:
#             try:
#                 gfld_path     = os.path.join(p.s3_proj_dir, 'aligned_rasters', f'gfld_1km_{year}.tif')
#                 sdr_path      = os.path.join(p.s3_proj_dir, 'preprocessed_rasters', f'avoided_export_{year}_filled.tif')
#                 era5_path     = os.path.join(p.s3_proj_dir, 'preprocessed_rasters', f'ERA5_annual_precip_{year}_filled.tif')
#                 worldpop_path = os.path.join(p.s3_proj_dir, 'preprocessed_rasters', f'ppp_{year}_1km_Aggregated_filled.tif')

#                 arrays, ndvs, failed = {}, {}, False
#                 for name, path in [('gfld', gfld_path), ('sdr', sdr_path),
#                                     ('era5', era5_path), ('worldpop', worldpop_path)]:
#                     ds = gdal.Open(path)
#                     if ds is None:
#                         hb.log(f'  Year {year}: Could not open {path}, skipping year')
#                         failed = True
#                         break
#                     band = ds.GetRasterBand(1)
#                     arrays[name] = band.ReadAsArray(col_offset, row_offset, n_cols, n_rows).astype(np.float64)
#                     ndvs[name] = band.GetNoDataValue()
#                     ds = None
#                 if failed:
#                     continue

#             except Exception as e:
#                 hb.log(f'  Year {year}: Error loading rasters - {str(e)[:100]}')
#                 continue

#             # Valid mask — only gfld can have meaningful NDV here since
#             # sdr/era5/worldpop have been gap-filled
#             valid_mask = np.ones((n_rows, n_cols), dtype=bool)
#             for name in ['gfld', 'sdr', 'era5', 'worldpop']:
#                 ndv = ndvs[name]
#                 arr = arrays[name]
#                 if ndv is not None:
#                     atol = abs(ndv) * 1e-5 if ndv != 0 else 1e-10
#                     valid_mask &= ~np.isclose(arr, ndv, rtol=0, atol=atol)
#                 valid_mask &= np.isfinite(arr)
#             valid_mask &= (arrays['gfld'] >= 0)
#             valid_mask &= (arrays['worldpop'] >= 0)

#             if np.sum(valid_mask) == 0:
#                 hb.log(f'  Year {year}: No valid pixels, skipping')
#                 continue

#             year_df = pd.DataFrame({
#                 'gfld':     arrays['gfld'][valid_mask],
#                 'sdr':      arrays['sdr'][valid_mask],
#                 'era5':     arrays['era5'][valid_mask],
#                 'worldpop': arrays['worldpop'][valid_mask],
#                 'year':     year
#             })
#             all_year_dfs.append(year_df)
#             del arrays
#             gc.collect()

#         # ------------------------------------------------------------------
#         # Helper to write a skipped result
#         # ------------------------------------------------------------------
#         def _write_skipped(reason):
#             rows = []
#             for part in ['logit', 'gamma']:
#                 for var in ['Intercept', 'sdr_z', 'era5_z', 'worldpop_z']:
#                     rows.append({
#                         'tile_name': tile_name, 'model_part': part, 'variable': var,
#                         'coefficient': np.nan, 'se': np.nan, 't_stat': np.nan,
#                         'p_value': np.nan, 'n_obs_total': 0, 'n_obs_positive': 0,
#                         'aic_logit': np.nan, 'aic_gamma': np.nan,
#                         'skipped': True, 'skip_reason': reason,
#                         'sdr_mean': np.nan, 'sdr_std': np.nan,
#                         'era5_mean': np.nan, 'era5_std': np.nan,
#                         'worldpop_mean': np.nan, 'worldpop_std': np.nan,
#                     })
#             pd.DataFrame(rows).to_csv(results_path, index=False)
#             hb.log(f'Tile {tile_name}: skipped - {reason}')

#         if len(all_year_dfs) == 0:
#             _write_skipped('no_valid_years')
#             return

#         panel_df = pd.concat(all_year_dfs, ignore_index=True)
#         n_obs_total = len(panel_df)
#         n_obs_positive = int((panel_df['gfld'] > 0).sum())
#         hb.log(f'Tile {tile_name}: {n_obs_total:,} total obs, '
#                f'{n_obs_positive:,} positive ({100*n_obs_positive/n_obs_total:.3f}%)')

#         if n_obs_total < 50:
#             _write_skipped(f'insufficient_obs_{n_obs_total}')
#             return
#         if n_obs_positive < 10:
#             _write_skipped(f'insufficient_positive_obs_{n_obs_positive}')
#             return

#         # ------------------------------------------------------------------
#         # Standardize predictors, save scaling params
#         # ------------------------------------------------------------------
#         scaling = {}
#         for var in ['sdr', 'era5', 'worldpop']:
#             mu = panel_df[var].mean()
#             sd = panel_df[var].std()
#             if sd < 1e-10:
#                 _write_skipped(f'zero_variance_{var}')
#                 return
#             panel_df[f'{var}_z'] = (panel_df[var] - mu) / sd
#             scaling[f'{var}_mean'] = mu
#             scaling[f'{var}_std'] = sd

#         exog_vars = ['sdr_z', 'era5_z', 'worldpop_z']
#         exog = sm.add_constant(panel_df[exog_vars], has_constant='add')

#         # ------------------------------------------------------------------
#         # Part 1: Logistic regression — P(gfld > 0)
#         # ------------------------------------------------------------------
#         binary_y = (panel_df['gfld'] > 0).astype(int)
#         logit_result = None
#         try:
#             with warnings.catch_warnings():
#                 warnings.simplefilter('ignore')
#                 logit_result = Logit(binary_y, exog).fit(
#                     method='bfgs', maxiter=200, disp=False
#                 )
#             hb.log(f'  Logit: converged={logit_result.mle_retvals.get("converged","?")}, '
#                    f'AIC={logit_result.aic:.2f}')
#         except Exception as e:
#             _write_skipped(f'logit_failed_{str(e)[:80]}')
#             return

#         # ------------------------------------------------------------------
#         # Part 2: Gamma GLM — E[gfld | gfld > 0]
#         # ------------------------------------------------------------------
#         positive_df = panel_df[panel_df['gfld'] > 0].copy()
#         exog_pos = sm.add_constant(positive_df[exog_vars], has_constant='add')
#         gamma_result = None
#         try:
#             with warnings.catch_warnings():
#                 warnings.simplefilter('ignore')
#                 gamma_result = sm.GLM(
#                     positive_df['gfld'],
#                     exog_pos,
#                     family=sm.families.Gamma(link=sm.families.links.Log())
#                 ).fit(method='irls', maxiter=200, disp=False)
#             hb.log(f'  Gamma: converged={gamma_result.converged}, '
#                    f'AIC={gamma_result.aic:.2f}')
#         except Exception as e:
#             _write_skipped(f'gamma_failed_{str(e)[:80]}')
#             return

#         # ------------------------------------------------------------------
#         # Store results — one row per model_part x variable
#         # ------------------------------------------------------------------
#         rows = []
#         for model_part, result in [('logit', logit_result), ('gamma', gamma_result)]:
#             coefficients = result.params
#             se_coef  = result.bse    if hasattr(result, 'bse')    else pd.Series(dtype=float)
#             t_stats  = result.tvalues if hasattr(result, 'tvalues') else pd.Series(dtype=float)
#             p_values = result.pvalues if hasattr(result, 'pvalues') else pd.Series(dtype=float)
#             aic_logit = logit_result.aic
#             aic_gamma = gamma_result.aic

#             for raw_name, friendly in [('const', 'Intercept'),
#                                         ('sdr_z',      'sdr_z'),
#                                         ('era5_z',     'era5_z'),
#                                         ('worldpop_z', 'worldpop_z')]:
#                 rows.append({
#                     'tile_name':      tile_name,
#                     'model_part':     model_part,
#                     'variable':       friendly,
#                     'coefficient':    coefficients.get(raw_name, np.nan),
#                     'se':             se_coef.get(raw_name, np.nan),
#                     't_stat':         t_stats.get(raw_name, np.nan),
#                     'p_value':        p_values.get(raw_name, np.nan),
#                     'n_obs_total':    n_obs_total,
#                     'n_obs_positive': n_obs_positive,
#                     'aic_logit':      aic_logit,
#                     'aic_gamma':      aic_gamma,
#                     'skipped':        False,
#                     'skip_reason':    None,
#                     **scaling
#                 })

#         pd.DataFrame(rows).to_csv(results_path, index=False)
#         hb.log(f'Saved results to: {results_path}')

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

# def aggregate_damage_fn_coefficients(p):
#     """
#     Aggregate hurdle model coefficients for a single tile.
#     Groups by (model_part, variable) and averages statistics.
#     Scaling params (tile-level constants) are passed through from the first valid row.
#     Writes tile_damage_function_results_agg.csv for use by generate_tile_avoided_mortality.
#     Always writes the file, even if skipped, so downstream tasks never crash on missing file.
#     """
#     if not p.run_this:
#         return
    
#     # Use tile base directory, not p.cur_dir
#     tile_base_dir = p.cur_dir_parent_dir
#     results_dir = os.path.join(tile_base_dir, 'aggregate_damage_fn_coefficients')
#     os.makedirs(results_dir, exist_ok=True)

#     # results_path = os.path.join(p.cur_dir, 'tile_damage_function_results_agg.csv')
#     results_path = os.path.join(results_dir, 'tile_damage_function_results_agg.csv')
#     if os.path.exists(results_path):
#         return

#     def _write_skipped(reason):
#         hb.log(f'  Writing skipped sentinel: {reason}')
#         pd.DataFrame([{'model_part': None, 'variable': None, 'coefficient': np.nan,
#                        'se': np.nan, 't_stat': np.nan, 'p_value': np.nan,
#                        'n_obs_total': 0, 'n_obs_positive': 0,
#                        'aic_logit': np.nan, 'aic_gamma': np.nan,
#                        'sdr_mean': np.nan, 'sdr_std': np.nan,
#                        'era5_mean': np.nan, 'era5_std': np.nan,
#                        'worldpop_mean': np.nan, 'worldpop_std': np.nan,
#                        'skipped': True}]).to_csv(results_path, index=False)

#     # Read from estimate_damage_fn sibling directory
#     estimate_dir = os.path.join(tile_base_dir, 'estimate_damage_fn')
#     tile_results_path = os.path.join(estimate_dir, 'tile_damage_function_results.csv')
#     # tile_results_path = os.path.join(p.estimate_damage_fn_dir, 'tile_damage_function_results.csv')
#     if not os.path.exists(tile_results_path):
#         hb.log(f'Warning: Tile results not found: {tile_results_path}')
#         _write_skipped('tile_results_not_found')
#         return

#     tile_results = pd.read_csv(tile_results_path)
#     valid = tile_results[
#         (tile_results['skipped'] == False) &
#         (tile_results['coefficient'].notnull())
#     ]

#     if valid.empty:
#         hb.log(f'No valid coefficients to aggregate at {tile_results_path}')
#         _write_skipped('no_valid_coefficients')
#         return

#     # Group by model_part and variable — one row per (logit/gamma x variable)
#     agg_cols = {
#         'coefficient':    'mean',
#         'se':             'mean',
#         't_stat':         'mean',
#         'p_value':        'mean',
#         'n_obs_total':    'first',
#         'n_obs_positive': 'first',
#         'aic_logit':      'first',
#         'aic_gamma':      'first',
#     }
#     avg_df = valid.groupby(['model_part', 'variable']).agg(agg_cols).reset_index()

#     scaling_cols = ['sdr_mean', 'sdr_std', 'era5_mean', 'era5_std', 'worldpop_mean', 'worldpop_std']
#     first_row = valid.iloc[0]
#     for col in scaling_cols:
#         avg_df[col] = first_row[col] if col in first_row.index else np.nan

#     avg_df['skipped'] = False
#     avg_df.to_csv(results_path, index=False)

#     n_obs = avg_df['n_obs_total'].iloc[0]
#     n_pos = avg_df['n_obs_positive'].iloc[0]
#     aic_l = avg_df['aic_logit'].iloc[0]
#     aic_g = avg_df['aic_gamma'].iloc[0]
#     hb.log(f'Aggregated hurdle coefficients saved to: {results_path}')
#     hb.log(f'  n_obs={n_obs:,}, n_positive={n_pos:,}, AIC_logit={aic_l:.2f}, AIC_gamma={aic_g:.2f}')
#     for part in ['logit', 'gamma']:
#         part_df = avg_df[avg_df['model_part'] == part]
#         hb.log(f'  [{part}]')
#         for _, row in part_df.iterrows():
#             hb.log(f'    {row["variable"]:15s}: {row["coefficient"]:10.6f} '
#                    f'(SE: {row["se"]:.6f}, p: {row["p_value"]:.4f})')


# def generate_tile_avoided_mortality(p):
#     if not p.run_this:
#         return

#     col_offset = int(p.tile_col_offset)
#     row_offset = int(p.tile_row_offset)
#     n_cols = int(p.tile_n_cols)
#     n_rows = int(p.tile_n_rows)
#     tile_name = f'{row_offset}_{col_offset}'
    
#     # Get the tile base directory (set by iterator)
#     tile_base_dir = p.cur_dir_parent_dir
    
#     # Construct this task's directory
#     task_dir = os.path.join(tile_base_dir, 'generate_tile_avoided_mortality')
#     os.makedirs(task_dir, exist_ok=True)
    
#     output_path = os.path.join(task_dir, f'avoided_mortality_tile_{tile_name}.npy')
#     if os.path.exists(output_path):
#         return
    
#     # Read from sibling task directory
#     agg_coef_dir = os.path.join(tile_base_dir, 'aggregate_damage_fn_coefficients')
#     agg_coef_path = os.path.join(agg_coef_dir, 'tile_damage_function_results_agg.csv')

#     coef_df = pd.read_csv(agg_coef_path)
#     coef_df['skipped'] = coef_df['skipped'].apply(
#         lambda x: x if isinstance(x, bool) else str(x).strip() == 'True'
#     )

#     if coef_df['skipped'].all():
#         hb.log(f'  Tile {tile_name}: BAIL - all skipped')
#         return

#     coef_df = coef_df[~coef_df['skipped']]

#     # Extract per-part coefficient series, indexed by variable name
#     def _get_coefs(part):
#         return coef_df[coef_df['model_part'] == part].set_index('variable')['coefficient']

#     logit_coefs = _get_coefs('logit')
#     gamma_coefs = _get_coefs('gamma')

#     if logit_coefs.empty or gamma_coefs.empty:
#         hb.log(f'  Missing logit or gamma coefficients for tile {tile_name}, skipping.')
#         return

#     # Scaling params — same on every row, read from first row
#     meta = coef_df.iloc[0]
#     scaling = {
#         'sdr':      (meta['sdr_mean'],      meta['sdr_std']),
#         'era5':     (meta['era5_mean'],     meta['era5_std']),
#         'worldpop': (meta['worldpop_mean'], meta['worldpop_std']),
#     }

#     # ------------------------------------------------------------------
#     # Load 2019 predictor rasters (windowed tile read)
#     # ------------------------------------------------------------------
#     sdr_path      = os.path.join(p.preprocess_aligned_rasters_dir, 'avoided_export_2019_filled.tif')
#     era5_path     = os.path.join(p.preprocess_aligned_rasters_dir, 'ERA5_annual_precip_2019_filled.tif')
#     worldpop_path = os.path.join(p.preprocess_aligned_rasters_dir, 'ppp_2019_1km_Aggregated_filled.tif')

#     try:
#         sdr_arr,      ndv_sdr      = _read_tile(sdr_path,      col_offset, row_offset, n_cols, n_rows)
#         era5_arr,     ndv_era5     = _read_tile(era5_path,     col_offset, row_offset, n_cols, n_rows)
#         worldpop_arr, ndv_worldpop = _read_tile(worldpop_path, col_offset, row_offset, n_cols, n_rows)
#     except Exception as e:
#         hb.log(f'  Tile {tile_name}: Error loading 2019 rasters - {str(e)}')
#         return

#     # ------------------------------------------------------------------
#     # Valid mask — after gap-filling, missing pixels are genuinely absent
#     # ------------------------------------------------------------------
#     valid_mask = (
#         ~_nodata_mask(sdr_arr,      ndv_sdr)      &
#         ~_nodata_mask(era5_arr,     ndv_era5)     &
#         ~_nodata_mask(worldpop_arr, ndv_worldpop) &
#         (worldpop_arr >= 0)
#     )
#     n_valid = int(np.sum(valid_mask))
#     hb.log(f'  Tile {tile_name}: {n_valid:,} valid pixels for prediction')

#     if n_valid == 0:
#         hb.log(f'  Tile {tile_name}: No valid pixels, skipping.')
#         return

#     # ------------------------------------------------------------------
#     # Standardize using training-time scaling params
#     # ------------------------------------------------------------------
#     def _standardize(arr, var):
#         mu, sd = scaling[var]
#         return (arr[valid_mask] - mu) / sd

#     sdr_z      = _standardize(sdr_arr,      'sdr')
#     era5_z     = _standardize(era5_arr,     'era5')
#     worldpop_z = _standardize(worldpop_arr, 'worldpop')

#     # Design matrix for valid pixels: [intercept, sdr_z, era5_z, worldpop_z]
#     ones = np.ones(n_valid)
#     X = np.column_stack([ones, sdr_z, era5_z, worldpop_z])
#     var_order = ['Intercept', 'sdr_z', 'era5_z', 'worldpop_z']

#     logit_b = np.array([logit_coefs.get(v, 0.0) for v in var_order])
#     gamma_b = np.array([gamma_coefs.get(v, 0.0) for v in var_order])

#     # ------------------------------------------------------------------
#     # Hurdle prediction: E[y] = P(y>0) * E[y | y>0]
#     # ------------------------------------------------------------------
#     # Part 1: logistic sigmoid -> P(gfld > 0)
#     eta_logit = np.clip(X @ logit_b, -50, 50)
#     p_nonzero = 1.0 / (1.0 + np.exp(-eta_logit))

#     # Part 2: Gamma log link -> E[gfld | gfld > 0]
#     eta_gamma = np.clip(X @ gamma_b, -50, 50)
#     mu_positive = np.exp(eta_gamma)

#     # Combined expected value
#     expected_mortality = (p_nonzero * mu_positive).astype(np.float32)

#     # ------------------------------------------------------------------
#     # Write output
#     # ------------------------------------------------------------------
#     out_arr = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
#     out_arr[valid_mask] = expected_mortality

#     hb.log(f'  Tile {tile_name}: predicted min={np.nanmin(out_arr):.6f}, '
#            f'max={np.nanmax(out_arr):.6f}, mean={np.nanmean(out_arr):.6f}')

#     np.save(output_path, out_arr)
#     hb.log(f'  Saved: {output_path}')

#     del sdr_arr, era5_arr, worldpop_arr, out_arr
#     gc.collect()

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
