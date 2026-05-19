import os
import sys
import json
import gc
import shutil

import hazelbean as hb
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from osgeo import gdal, ogr, osr
import pygeoprocessing as pygeo
import statsmodels.api as sm

from .utils import s3_handler

# Configure GDAL to limit open file handles and improve S3 handling
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'YES')
gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')
gdal.SetConfigOption('GDAL_CACHEMAX', '512')  # Cache up to 512 MB
gdal.SetConfigOption('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', 'tif,tiff')
gdal.SetConfigOption('CPL_VSIL_CURL_NON_CACHED', 'NO')  # Enable caching for AWS S3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_band(path):
    """Open a single-band raster. Returns (dataset, band, ndv). Raises on failure."""
    # Check if path is already local (absolute path or exists as file)
    if os.path.isabs(path) or os.path.exists(path):
        s3_path = path
    else:
        # Convert relative path to S3
        s3_path = s3_handler.get_vsis3_path(path)
    
    ds = gdal.Open(s3_path)
    if ds is None:
        raise FileNotFoundError(f'GDAL could not open: {s3_path}')
    band = ds.GetRasterBand(1)
    ndv  = band.GetNoDataValue()
    return ds, band, ndv


def _close_gdal_dataset(ds):
    """Properly close a GDAL dataset and release resources."""
    if ds is not None:
        try:
            ds.FlushCache()
        except:
            pass
        ds = None
    gc.collect()  # Force garbage collection


def _safe_read_array(band, dtype, path_hint=''):
    """Safely read array from GDAL band, with validation. Raises on failure."""
    arr = band.ReadAsArray()
    if arr is None:
        raise RuntimeError(f'Failed to read array from band{" (" + path_hint + ")" if path_hint else ""}: '
                          f'ReadAsArray() returned None. File may be corrupted or S3 read failed.')
    try:
        return arr.astype(dtype)
    except Exception as e:
        raise RuntimeError(f'Failed to convert array to {dtype}{" (" + path_hint + ")" if path_hint else ""}: {e}')


def _log_array_memory(name, arr):
    """Log array shape/dtype/size to diagnose RAM spikes."""
    try:
        nbytes_gb = float(arr.nbytes) / (1024.0 ** 3)
        hb.log(f'      {name}: shape={arr.shape}, dtype={arr.dtype}, size={nbytes_gb:.2f} GB')
    except Exception:
        hb.log(f'      {name}: <unable to report memory info>')


def _get_or_cache_local_path(original_path, cache_dir, max_retries=3):
    """
    For S3 paths, download to local cache. For local paths, return as-is.
    This avoids repeated S3 reads which can fail with ZSTD compression.
    
    Args:
        original_path: Path like 'Files/base_data/.../file.tif' or S3 vsis3 path
        cache_dir: Local directory to cache files
        max_retries: Retry attempts for download
    
    Returns:
        Local file path to use (either original local path or cached copy)
    """
    # If not an S3 path, return original
    if not original_path.startswith('/vsis3') and not os.path.expanduser('~') in original_path:
        # Check if it's a relative path like "Files/base_data/.../file.tif"
        full_local_path = os.path.join(os.path.expanduser('~'), original_path)
        if os.path.exists(full_local_path):
            return full_local_path
    
    # For local paths that exist, return as-is
    if os.path.exists(original_path):
        return original_path
    
    # If original is a relative path, attempt to find it locally first
    if not original_path.startswith('/vsis3'):
        full_path = os.path.join(os.path.expanduser('~'), original_path)
        if os.path.exists(full_path):
            return full_path
    
    # Delegate S3 downloads to the shared S3Handler implementation.
    cache_filename = os.path.basename(original_path)
    hb.log(f'    Caching to local: {cache_filename}')
    cache_path = s3_handler.get_or_cache_local_path(
        original_path,
        cache_dir,
        max_retries=max_retries,
    )
    hb.log(f'    Cached successfully: {cache_filename}')
    return cache_path


def _get_cached_estimation_table_path(original_path, cache_root, year=None, prefer_year=True):
    """Return a path from the estimation-table cache if it exists."""
    basename = os.path.basename(original_path)
    candidates = []
    if prefer_year and year is not None:
        candidates.append(os.path.join(cache_root, f'year_{year}', basename))
    candidates.append(os.path.join(cache_root, basename))
    if not prefer_year and year is not None:
        candidates.append(os.path.join(cache_root, f'year_{year}', basename))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _read_pixels(path, rows, cols, expected_shape=None):
    """
    Extract values at (rows, cols) index arrays from a single-band raster.
    Returns a float32 array of length len(rows), with nodata replaced by NaN.

    Loads the full array into RAM — appropriate for global 1 km grids
    (~600 MB worst case for float32, well within typical limits).
    Uses plain GDAL ReadAsArray; no rasterio.
    """
    ds, band, ndv = _open_band(path)
    arr = _safe_read_array(band, np.float32, path)
    _close_gdal_dataset(ds)  # Properly close the dataset

    # Trim off-by-one edges if raster is slightly oversized vs reference
    if expected_shape is not None:
        if arr.shape != expected_shape:
            hb.log(f'  Shape mismatch in {os.path.basename(path)}: '
                   f'{arr.shape} vs expected {expected_shape} — trimming.')
            arr = arr[:expected_shape[0], :expected_shape[1]]

    vals = arr[rows, cols]
    if ndv is not None:
        vals[vals == np.float32(ndv)] = np.nan
    return vals



import rasterio
from rasterio.transform import xy as rio_xy

def _get_openable_path(path):
    if os.path.isabs(path) or os.path.exists(path):
        return path
    return s3_handler.get_vsis3_path(path)


# AEZ grouping based on the class labels before the semicolon in the SLD legend.
# This keeps the model much more expressive than the previous 6-bin collapse,
# while still avoiding 57 nearly-saturated fixed effects.
AEZ_GROUP_RANGES = [
    (1, 6),    # Tropics lowland
    (7, 12),   # Tropics highland
    (13, 18),  # Sub-tropics warm
    (19, 24),  # Sub-tropics mod. cool
    (25, 30),  # Sub-tropics cool
    (31, 36),  # Temperate moderate
    (37, 42),  # Temperate cool
    (43, 48),  # Cold no permafrost
    (49, 49),  # Dominantly very steep terrain
    (50, 50),  # Land with severe soil/terrain limitations
    (51, 51),  # Ample irrigated soils
    (52, 52),  # Dominantly hydromorphic soils
    (53, 53),  # Desert/Arid climate
    (54, 54),  # Boreal/Cold climate
    (55, 55),  # Arctic/Very cold climate
    (56, 56),  # Dominantly built-up land
    (57, 57),  # Dominantly water
]

def _map_zone_to_group_array():
    """Return a lookup array mapping 0..57 -> group id (1..len(AEZ_GROUP_RANGES)), 0 for invalid."""
    max_zone = 57
    lookup = np.zeros(max_zone + 1, dtype=np.int16)
    for gid, (a, b) in enumerate(AEZ_GROUP_RANGES, start=1):
        lookup[a:b+1] = gid
    return lookup

_GAEZ_TO_GROUP = _map_zone_to_group_array()

def _zone_to_group(z):
    try:
        z_int = int(z)
    except Exception:
        return 0
    if 0 <= z_int < len(_GAEZ_TO_GROUP):
        return int(_GAEZ_TO_GROUP[z_int])
    return 0

def _sample_raster_values(path, rows, cols, expected_shape=None, chunk=100000):
    """Return float32 array of sampled values at (rows,cols) using rasterio.sample (batched)."""
    path = _get_openable_path(path)
    vals_list = []
    with rasterio.open(path) as src:
        # Optionally check/trim expected shape
        if expected_shape is not None:
            if (src.height, src.width) != (expected_shape[0], expected_shape[1]):
                hb.log(f'  Shape mismatch in {os.path.basename(path)}: '
                       f'({src.height},{src.width}) vs expected {expected_shape} — treating samples carefully.')

        nodata = src.nodatavals[0] if src.nodatavals else None
        transform = src.transform

        # Convert rows,cols to coordinates (center)
        # rasterio.transform.xy supports arrays
        xs, ys = rio_xy(transform, rows, cols, offset='center')
        coords = list(zip(xs, ys))

        # sample in chunks to avoid large memory spikes
        for i in range(0, len(coords), chunk):
            chunk_coords = coords[i:i+chunk]
            samples = list(src.sample(chunk_coords))
            # samples: list of arrays shape (bands,) ; single-band -> pick [0]
            arr = np.fromiter((s[0] for s in samples), dtype=np.float32, count=len(samples))
            if nodata is not None:
                arr[arr == np.float32(nodata)] = np.nan
            vals_list.append(arr)

    if vals_list:
        return np.concatenate(vals_list)
    return np.empty(0, dtype=np.float32)

    

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def build_estimation_table(p):
    """
    Build a pixel-year panel for hazard model estimation using stratified
    case-control sampling.

    Strategy
    --------
    EVENTS   — every pixel-year where a UGLC landslide was recorded.
               Covariates are extracted at each event location and year.

    CONTROLS — non-event pixel-years sampled at control_ratio (default 50)
               times the number of events per year, stratified by FAO GAEZ
               zone.  Stratification ensures the regression sees the full
               landscape diversity rather than being dominated by flat, dry,
               low-slope areas that make up most of the globe.

    Intercept correction
    --------------------
    Because controls are under-sampled relative to reality, the fitted
    logit intercept will be biased downward.  The King & Zeng (2001)
    correction adds log(1 / sampling_fraction) to the intercept, where:

        sampling_fraction = n_controls_sampled / n_total_non_event_pixel_years

    This value is saved to sampling_meta.csv so estimate_hazard_model can
    apply it without reloading any rasters.

    Time-invariant covariates
    -------------------------
    slope_degree        slope in degrees     (Geomorpho90m)
    roughness           terrain roughness     (Geomorpho90m)
    tpi                 topographic position  (Geomorpho90m)
    elev_stdev          elevation std. dev.   (Geomorpho90m)
    road_density        km road / km²        (GRIP4)
    dist_to_fault_km    km to nearest active fault (GEM)
    gaez_zone           FAO GAEZ zone        (strata + model fixed effect)

    Annual covariates  (extracted per pixel-year)
    ---------------------------------------------
    forest_share         esacci_share_forest_{year}.tif
    cropland_share       esacci_share_cropland_{year}.tif
    grassland_share      esacci_share_grassland_{year}.tif
    urban_share          esacci_share_urban_{year}.tif
    water_share          esacci_share_water_{year}.tif
    othernat_share       esacci_share_othernat_{year}.tif
    other_share          esacci_share_other_{year}.tif
    rain_max_daily       era5_max_daily_mm_{year}.tif
    population           landscan_{year}_1km.tif
    avoided_erosion_obs  preprocess_data/sdr/avoided_erosion_observed_{year}.tif
    avoided_erosion_cf   preprocess_data/sdr/avoided_erosion_counterfactual_{year}.tif

    Event-only covariates
    ---------------------
    mortality            uglc_mortality_{year}.tif   (0.0 for control rows)

    Outputs
    -------
    {cur_dir}/estimation_table.parquet
    {cur_dir}/sampling_meta.csv
    """
    if not p.run_this:
        return p

    out_parquet = os.path.join(p.cur_dir, 'estimation_table.parquet')
    out_meta    = os.path.join(p.cur_dir, 'sampling_meta.csv')

    event_source = 'uglc'

    if os.path.exists(out_parquet) and os.path.exists(out_meta):
        hb.log('Estimation table already exists, skipping.')
        return p

    control_ratio = getattr(p, 'control_ratio', 100)
    rng           = np.random.default_rng(seed=42)
    # minimum fraction of controls that should be populated; warn if lower
    min_pop_control_frac = getattr(p, 'min_pop_control_frac', 0.05)

    lulc_vars = [
        'forest', 'cropland', 'grassland', 'urban',
        'water', 'othernat', 'other',
    ]
    # Refined ESA categories (added to estimation table as additional covariates)
    refined_lulc_vars = [
        'forest_dense', 'forest_open',
        'cropland_rainfed', 'cropland_irrigated',
        'grassland', 'shrubland', 'sparse', 'bare_areas',
    ]
    # Deforestation exposures derived from ESA shares.
    deforestation_specs = [
        ('deforestation_forest_share_1yr', 'coarse', 'forest_share', 1),
        ('deforestation_forest_share_3yr', 'coarse', 'forest_share', 3),
    ]

    # ------------------------------------------------------------------
    # 1. Load time-invariant rasters once
    # ------------------------------------------------------------------
    # Geomorpho90m terrain variables (preferred over GEDTM where available)
    slope_path  = os.path.join(*p.base_data, 'preprocess_data', 'geomorpho90m',
                               'slope_geomorpho.tif')
    roughness_path = os.path.join(*p.base_data, 'preprocess_data', 'geomorpho90m',
                                   'roughness_geomorpho.tif')
    tpi_path    = os.path.join(*p.base_data, 'preprocess_data', 'geomorpho90m',
                               'tpi_geomorpho.tif')
    elev_stdev_path = os.path.join(*p.base_data, 'preprocess_data', 'geomorpho90m',
                                    'elev_stdev_geomorpho.tif')
    
    # Legacy GEDTM variables (LS-factor, TWI; slope replaced by Geomorpho)
    # ls_path     = os.path.join(*p.base_data, 'preprocess_data', 'gedtm',
    #                            'lsfactor_gedtm.tif')
    # twi_path    = os.path.join(*p.base_data, 'preprocess_data', 'gedtm',
    #                            'twi_gedtm.tif')
    gaez_path   = os.path.join(*p.base_data, 'preprocess_data', 'fao_gaez',
                               'fao_gaez.tif')
    roads_path  = os.path.join(*p.base_data, 'preprocess_data', 'grip_roads',
                               'road_density_km_per_km2.tif')
    faults_path = os.path.join(*p.base_data, 'preprocess_data', 'gem_faults',
                               'distance_to_fault_km.tif')
    motorized_travel_path = os.path.join(*p.base_data, 'preprocess_data', 'malariaatlas',
                                         'motorized_travel_time_to_healthcare.tif')
    walking_travel_path   = os.path.join(*p.base_data, 'preprocess_data', 'malariaatlas',
                                         'walking_only_travel_time_to_healthcare.tif')

    # Setup local cache for static rasters to avoid repeated S3 reads with ZSTD compression
    cache_dir = os.path.join(p.cur_dir, 'static_raster_cache')
    
    hb.log('Loading time-invariant rasters...')
    hb.log('  Caching static rasters locally...')
    slope_path_local  = _get_or_cache_local_path(slope_path, cache_dir)
    roughness_path_local = _get_or_cache_local_path(roughness_path, cache_dir) if s3_handler.file_exists(roughness_path) else None
    tpi_path_local    = _get_or_cache_local_path(tpi_path, cache_dir) if s3_handler.file_exists(tpi_path) else None
    elev_stdev_path_local = _get_or_cache_local_path(elev_stdev_path, cache_dir) if s3_handler.file_exists(elev_stdev_path) else None
    # ls_path_local     = _get_or_cache_local_path(ls_path, cache_dir)
    # twi_path_local    = _get_or_cache_local_path(twi_path, cache_dir)
    gaez_path_local   = _get_or_cache_local_path(gaez_path, cache_dir)
    roads_path_local  = _get_or_cache_local_path(roads_path, cache_dir) if s3_handler.file_exists(roads_path) else None
    faults_path_local = _get_or_cache_local_path(faults_path, cache_dir) if s3_handler.file_exists(faults_path) else None
    
    hb.log('  Loading cached rasters...')
    ds_slope,  band_slope,  _        = _open_band(slope_path_local)
    ds_roughness, band_roughness, _ = _open_band(roughness_path_local) if roughness_path_local is not None else (None, None, None)
    ds_tpi,    band_tpi,    _        = _open_band(tpi_path_local) if tpi_path_local is not None else (None, None, None)
    ds_elev_stdev, band_elev_stdev, _ = _open_band(elev_stdev_path_local) if elev_stdev_path_local is not None else (None, None, None)
    # ds_ls,     band_ls,     _        = _open_band(ls_path_local)
    # ds_twi,    band_twi,    _        = _open_band(twi_path_local)
    ds_gaez,   band_gaez,   gaez_ndv = _open_band(gaez_path_local)

    slope_arr = _safe_read_array(band_slope, np.float32, slope_path_local)
    roughness_arr = _safe_read_array(band_roughness, np.float32, roughness_path_local) if band_roughness is not None else None
    tpi_arr = _safe_read_array(band_tpi, np.float32, tpi_path_local) if band_tpi is not None else None
    elev_stdev_arr = _safe_read_array(band_elev_stdev, np.float32, elev_stdev_path_local) if band_elev_stdev is not None else None
    # ls_arr    = _safe_read_array(band_ls, np.float32, ls_path_local)
    # twi_arr   = _safe_read_array(band_twi, np.float32, twi_path_local)
    gaez_arr  = _safe_read_array(band_gaez, np.int16, gaez_path_local)

    # Diagnose raster sizes
    hb.log(f'  Raster shapes after load:')
    hb.log(f'    slope_arr: {slope_arr.shape}')
    if roughness_arr is not None:
        hb.log(f'    roughness_arr: {roughness_arr.shape}')
    if tpi_arr is not None:
        hb.log(f'    tpi_arr: {tpi_arr.shape}')
    if elev_stdev_arr is not None:
        hb.log(f'    elev_stdev_arr: {elev_stdev_arr.shape}')
    # hb.log(f'    ls_arr: {ls_arr.shape}')
    # hb.log(f'    twi_arr: {twi_arr.shape}')
    hb.log(f'    gaez_arr: {gaez_arr.shape}')
    
    # Properly close GDAL datasets to free file handles
    _close_gdal_dataset(ds_slope)
    if ds_roughness is not None:
        _close_gdal_dataset(ds_roughness)
    if ds_tpi is not None:
        _close_gdal_dataset(ds_tpi)
    if ds_elev_stdev is not None:
        _close_gdal_dataset(ds_elev_stdev)
    # _close_gdal_dataset(ds_ls)
    # _close_gdal_dataset(ds_twi)
    _close_gdal_dataset(ds_gaez)

    # Roads and faults are optional — log warning if missing but continue
    has_roads  = s3_handler.file_exists(roads_path)
    has_faults = s3_handler.file_exists(faults_path)

    if has_roads:
        ds_roads, band_roads, _ = _open_band(roads_path_local)
        roads_arr = _safe_read_array(band_roads, np.float32, roads_path_local)
        _close_gdal_dataset(ds_roads)  # Properly close
        hb.log('  Road density raster loaded.')
    else:
        roads_arr = None
        raise FileNotFoundError(f'Road density raster not found at {roads_path}.')

    if has_faults:
        ds_faults, band_faults, _ = _open_band(faults_path_local)
        faults_arr = _safe_read_array(band_faults, np.float32, faults_path_local)
        _close_gdal_dataset(ds_faults)  # Properly close
        hb.log('  Fault distance raster loaded.')
    else:
        faults_arr = None
        raise FileNotFoundError(f'Fault distance raster not found at {faults_path}.')

    # Load travel-time rasters (time-invariant infrastructure)
    has_motorized_travel = s3_handler.file_exists(motorized_travel_path)
    has_walking_travel   = s3_handler.file_exists(walking_travel_path)
    
    if has_motorized_travel:
        motorized_travel_path_local = _get_or_cache_local_path(motorized_travel_path, cache_dir)
        ds_mot, band_mot, _ = _open_band(motorized_travel_path_local)
        motorized_travel_arr = _safe_read_array(band_mot, np.float32, motorized_travel_path_local)
        _close_gdal_dataset(ds_mot)
        hb.log('  Motorized travel time raster loaded.')
    else:
        motorized_travel_arr = None
        raise FileNotFoundError(f'Motorized travel time raster not found at {motorized_travel_path}.')

    if has_walking_travel:
        walking_travel_path_local = _get_or_cache_local_path(walking_travel_path, cache_dir)
        ds_walk, band_walk, _ = _open_band(walking_travel_path_local)
        walking_travel_arr = _safe_read_array(band_walk, np.float32, walking_travel_path_local)
        _close_gdal_dataset(ds_walk)
        hb.log('  Walking travel time raster loaded.')
    else:
        walking_travel_arr = None
        raise FileNotFoundError(f'Walking travel time raster not found at {walking_travel_path}.')

    raster_size    = p.reference_raster_info['raster_size']
    n_cols, n_rows = raster_size[0], raster_size[1]
    ref_shape      = (n_rows, n_cols)

    slope_arr  = slope_arr[:ref_shape[0],  :ref_shape[1]]
    roughness_arr = roughness_arr[:ref_shape[0], :ref_shape[1]] if roughness_arr is not None else None
    tpi_arr    = tpi_arr[:ref_shape[0],    :ref_shape[1]] if tpi_arr is not None else None
    elev_stdev_arr = elev_stdev_arr[:ref_shape[0], :ref_shape[1]] if elev_stdev_arr is not None else None
    gaez_arr   = gaez_arr[:ref_shape[0],   :ref_shape[1]]
    if roads_arr  is not None:
        roads_arr  = roads_arr[:ref_shape[0],  :ref_shape[1]]
    if faults_arr is not None:
        faults_arr = faults_arr[:ref_shape[0], :ref_shape[1]]

    # Clip terrain rasters to valid physical range.
    slope_arr = np.where((slope_arr > 90) | (slope_arr < 0), np.nan, slope_arr)
    if roughness_arr is not None:
        roughness_arr = np.where(roughness_arr < 0, np.nan, roughness_arr)
    if elev_stdev_arr is not None:
        elev_stdev_arr = np.where(elev_stdev_arr < 0, np.nan, elev_stdev_arr)

    # ------------------------------------------------------------------
    # 2. Global land mask
    # ------------------------------------------------------------------
    land_mask = np.ones((n_rows, n_cols), dtype=bool)
    land_mask &= np.isfinite(slope_arr)
    if roughness_arr is not None:
        land_mask &= np.isfinite(roughness_arr)
    if tpi_arr is not None:
        land_mask &= np.isfinite(tpi_arr)
    if elev_stdev_arr is not None:
        land_mask &= np.isfinite(elev_stdev_arr)

    GAEZ_NODATA = {0, 255}
    if gaez_ndv is not None:
        GAEZ_NODATA.add(int(gaez_ndv))
    for fill in GAEZ_NODATA:
        land_mask &= (gaez_arr != np.int16(fill))
    land_mask &= (gaez_arr >= 1) & (gaez_arr <= 57)

    land_rows, land_cols_idx = np.where(land_mask)
    n_land_pixels = len(land_rows)
    hb.log(f'Land pixels available for sampling: {n_land_pixels:,}')

    unique_zones_present = np.unique(gaez_arr[land_rows, land_cols_idx])
    hb.log(f'GAEZ zones present after masking: '
           f'{sorted(unique_zones_present.tolist())}')

    gaez_at_land = gaez_arr[land_rows, land_cols_idx]
    unique_zones = np.unique(gaez_at_land)
    zone_idx     = {int(z): np.where(gaez_at_land == z)[0]
                    for z in unique_zones}
    hb.log(f'GAEZ zones: {len(unique_zones)}')

    # ------------------------------------------------------------------
    # 3. Loop over years
    # ------------------------------------------------------------------
    
    # Setup intermediate directory for year-by-year checkpointing
    intermediate_tables_dir = os.path.join(p.cur_dir, 'intermediate_tables')
    os.makedirs(intermediate_tables_dir, exist_ok=True)
    
    # Check which years have already been processed
    completed_years = set()
    for filename in os.listdir(intermediate_tables_dir):
        if filename.startswith('year_') and filename.endswith('_events.parquet'):
            year_str = filename.replace('year_', '').replace('_events.parquet', '')
            try:
                completed_years.add(int(year_str))
            except ValueError:
                pass
    
    if completed_years:
        hb.log(f'Found {len(completed_years)} previously completed years: '
               f'{sorted(completed_years)}')
    
    all_years = sorted(int(y) for y in p.time_range)
    if not all_years:
        raise ValueError('p.time_range is empty in build_estimation_table.')

    lag_years = int(getattr(p, 'deforestation_max_lag_years', 0))
    default_estimation_start = all_years[0] + max(0, lag_years)
    estimation_start_year = int(getattr(p, 'estimation_start_year', default_estimation_start))
    estimation_years = [y for y in all_years if y >= estimation_start_year]
    if not estimation_years:
        raise ValueError(
            f'No estimation years available after filtering with estimation_start_year={estimation_start_year}. '
            f'Available years: {all_years}'
        )
    hb.log(
        f'Estimation year window: {estimation_years[0]}-{estimation_years[-1]} '
        f'(lag_years={lag_years}, configured_start={estimation_start_year})'
    )

    n_total_non_event_pixel_years = 0

    for year in estimation_years:
        year_events_path = os.path.join(intermediate_tables_dir, f'year_{year}_events.parquet')
        year_controls_path = os.path.join(intermediate_tables_dir, f'year_{year}_controls.parquet')
        
        # Skip if already processed
        if year in completed_years:
            hb.log(f'Year {year}: already processed, skipping...')
            # Still need to load n_total_non_event_pixel_years
            if os.path.exists(year_events_path):
                df_ev = pd.read_parquet(year_events_path)
                n_events_this_year = len(df_ev)
            else:
                n_events_this_year = 0
            n_total_non_event_pixel_years += (n_land_pixels - n_events_this_year)
            continue

        uglc_binary_path = os.path.join(*p.base_data, 'preprocess_data',
                        'uglc', f'uglc_binary_{year}.tif')
        uglc_mort_path   = os.path.join(*p.base_data, 'preprocess_data',
                        'uglc', f'uglc_mortality_{year}.tif')
        pop_path        = os.path.join(*p.base_data, 'preprocess_data',
                           'landscan', f'landscan_{year}_1km.tif')
        rain_path       = os.path.join(*p.base_data, 'preprocess_data',
                                       'era5_land', f'era5_max_daily_mm_{year}.tif')
        sdr_obs_path    = os.path.join(*p.base_data, 'preprocess_data',
                                       'sdr', f'avoided_erosion_observed_{year}.tif')
        sdr_cf_path     = os.path.join(*p.base_data, 'preprocess_data',
                                       'sdr', f'avoided_erosion_counterfactual_{year}.tif')
        lulc_paths = {
            var: os.path.join(*p.base_data, 'preprocess_data', 'esa',
                              f'esacci_share_{var}_{year}.tif')
            for var in lulc_vars
        }
        refined_lulc_paths = {
            var: os.path.join(*p.base_data, 'preprocess_data', 'esa_refined',
                              f'esacci_share_refined_{var}_{year}.tif')
            for var in refined_lulc_vars
        }
        deforestation_paths = {
            col: os.path.join(
                *p.base_data,
                'preprocess_data',
                'deforestation',
                folder,
                f'deforestation_{source}_{window}yr_{year}.tif',
            )
            for col, folder, source, window in deforestation_specs
        }

        required_paths = (
            [uglc_binary_path, uglc_mort_path, pop_path, rain_path]
            + list(lulc_paths.values())
            + list(refined_lulc_paths.values())
            + list(deforestation_paths.values())
        )
        missing_required = [p_ for p_ in required_paths
                            if not s3_handler.file_exists(p_)]
        if missing_required:
            raise FileNotFoundError(f"Year {year}: missing required file(s): {missing_required}")

        has_sdr = (s3_handler.file_exists(sdr_obs_path) and
                   s3_handler.file_exists(sdr_cf_path))
        if not has_sdr:
            hb.log(f'Year {year}: SDR not found — avoided_erosion will be NaN')

        hb.log(f'Year {year}: caching annual rasters locally...')

        # Cache annual rasters locally for fast repeated access (fail-fast on download errors).
        year_cache_dir = os.path.join(cache_dir, f'year_{year}')
        os.makedirs(year_cache_dir, exist_ok=True)

        uglc_binary_path_local = s3_handler.get_or_cache_local_path(uglc_binary_path, year_cache_dir)
        uglc_mort_path_local   = s3_handler.get_or_cache_local_path(uglc_mort_path, year_cache_dir)
        pop_path_local        = s3_handler.get_or_cache_local_path(pop_path, year_cache_dir)
        rain_path_local       = s3_handler.get_or_cache_local_path(rain_path, year_cache_dir)

        if has_sdr:
            sdr_obs_path_local = s3_handler.get_or_cache_local_path(sdr_obs_path, year_cache_dir)
            sdr_cf_path_local  = s3_handler.get_or_cache_local_path(sdr_cf_path, year_cache_dir)

        lulc_paths_local = {
            var: s3_handler.get_or_cache_local_path(lulc_paths[var], year_cache_dir)
            for var in lulc_vars
        }
        refined_lulc_paths_local = {
            var: s3_handler.get_or_cache_local_path(refined_lulc_paths[var], year_cache_dir)
            for var in refined_lulc_vars
        }
        deforestation_paths_local = {
            col: s3_handler.get_or_cache_local_path(deforestation_paths[col], year_cache_dir)
            for col in deforestation_paths
        }

        hb.log(f'  Loading cached year {year} rasters into memory...')

        ds_uglc,  band_uglc,  _ = _open_band(uglc_binary_path_local)
        uglc_arr  = _safe_read_array(band_uglc, np.int8, uglc_binary_path_local)
        uglc_arr  = uglc_arr[:ref_shape[0], :ref_shape[1]]
        _close_gdal_dataset(ds_uglc)  # Properly close
        _log_array_memory('uglc_arr', uglc_arr)

        hb.log('  Annual covariates will be sampled from cached rasters (memory-safe mode).')

        # ---- 3a. Events ----------------------------------------------
        event_mask       = (uglc_arr == 1) & land_mask
        ev_rows, ev_cols = np.where(event_mask)
        n_events         = len(ev_rows)
        hb.log(f'  {n_events} event pixels')

        if n_events > 0:
            rain_vals = _sample_raster_values(
                rain_path_local, ev_rows, ev_cols, expected_shape=ref_shape, chunk=200000)
            pop_vals = _sample_raster_values(
                pop_path_local, ev_rows, ev_cols, expected_shape=ref_shape, chunk=200000)
            mort_vals = _sample_raster_values(
                uglc_mort_path_local, ev_rows, ev_cols, expected_shape=ref_shape, chunk=200000)
            if has_sdr:
                sdr_obs_vals = _sample_raster_values(
                    sdr_obs_path_local, ev_rows, ev_cols, expected_shape=ref_shape, chunk=200000)
                sdr_cf_vals = _sample_raster_values(
                    sdr_cf_path_local, ev_rows, ev_cols, expected_shape=ref_shape, chunk=200000)
            else:
                sdr_obs_vals = np.full(n_events, np.nan, dtype=np.float32)
                sdr_cf_vals  = np.full(n_events, np.nan, dtype=np.float32)
            lulc_vals = {
                var: _sample_raster_values(
                    lulc_paths_local[var], ev_rows, ev_cols, expected_shape=ref_shape, chunk=200000)
                for var in lulc_vars
            }
            refined_lulc_vals = {
                var: _sample_raster_values(
                    refined_lulc_paths_local[var], ev_rows, ev_cols, expected_shape=ref_shape, chunk=200000)
                for var in refined_lulc_vars
            }
            deforestation_vals = {
                col: _sample_raster_values(
                    deforestation_paths_local[col], ev_rows, ev_cols, expected_shape=ref_shape, chunk=200000)
                for col in deforestation_paths_local
            }

            slope_vals   = slope_arr[ev_rows,  ev_cols]
            roughness_vals = (roughness_arr[ev_rows, ev_cols] if roughness_arr is not None 
                              else np.full(n_events, np.nan, dtype=np.float32))
            tpi_vals     = (tpi_arr[ev_rows, ev_cols] if tpi_arr is not None
                            else np.full(n_events, np.nan, dtype=np.float32))
            elev_stdev_vals = (elev_stdev_arr[ev_rows, ev_cols] if elev_stdev_arr is not None
                               else np.full(n_events, np.nan, dtype=np.float32))
            # ls_vals      = ls_arr[ev_rows,     ev_cols]
            # twi_vals     = twi_arr[ev_rows,    ev_cols]
            gaez_vals    = gaez_arr[ev_rows,   ev_cols]
            roads_vals   = (roads_arr[ev_rows,  ev_cols]
                            if roads_arr  is not None
                            else np.full(n_events, np.nan, dtype=np.float32))
            faults_vals  = (faults_arr[ev_rows, ev_cols]
                            if faults_arr is not None
                            else np.full(n_events, np.nan, dtype=np.float32))
            motorized_travel_vals = (motorized_travel_arr[ev_rows, ev_cols]
                                     if motorized_travel_arr is not None
                                     else np.full(n_events, np.nan, dtype=np.float32))
            walking_travel_vals   = (walking_travel_arr[ev_rows, ev_cols]
                                     if walking_travel_arr is not None
                                     else np.full(n_events, np.nan, dtype=np.float32))
            
            # Save events for this year immediately
            event_batch_temp = pd.DataFrame({
                'year':                np.full(n_events, year, dtype=np.int16),
                'row':                 ev_rows.astype(np.int32),
                'col':                 ev_cols.astype(np.int32),
                'landslide':           np.ones(n_events, dtype=np.int8),
                'slope_degree':        slope_vals,
                'roughness':           roughness_vals,
                'tpi':                 tpi_vals,
                'elev_stdev':          elev_stdev_vals,
                # 'ls_factor':           ls_vals,
                # 'twi':                 twi_vals,
                'road_density':        roads_vals,
                'dist_to_fault_km':    faults_vals,
                'motorized_travel_time': motorized_travel_vals,
                'walking_travel_time':   walking_travel_vals,
                'rain_max_daily':      rain_vals,
                'population':          pop_vals,
                'is_populated':        (pop_vals > 0),
                'mortality':           mort_vals,
                'avoided_erosion_obs': sdr_obs_vals,
                'avoided_erosion_cf':  sdr_cf_vals,
                'gaez_zone':           gaez_vals.astype(np.int16),
            })
            for var in lulc_vars:
                event_batch_temp[f'{var}_share'] = lulc_vals[var]
            for var in refined_lulc_vars:
                event_batch_temp[f'{var}_share_refined'] = refined_lulc_vals[var]
            for col in deforestation_vals:
                event_batch_temp[col] = deforestation_vals[col]
            
            year_events_path = os.path.join(intermediate_tables_dir, f'year_{year}_events.parquet')
            event_batch_temp.to_parquet(year_events_path, index=False)
            hb.log(f'    Saved {len(event_batch_temp)} events to {os.path.basename(year_events_path)}')

        # ---- 3c. Controls (dual sampling) ---------------------
        # Sample M controls from ALL land pixels (unrestricted) for Stage 1
        # Sample M controls from POPULATED land pixels only (restricted) for Stage 2
        n_controls_this_year = n_events * control_ratio
        if n_controls_this_year == 0:
            del glc_arr
            gc.collect()
            continue

        # UNRESTRICTED CONTROLS: from all land pixels
        hb.log(f'  Sampling unrestricted controls (all pixels)...')
        sampled_land_indices = []
        for zone, idx in zone_idx.items():
            n_zone = max(1, round(n_controls_this_year * len(idx) / n_land_pixels))
            n_zone = min(n_zone, len(idx))
            sampled_land_indices.append(
                rng.choice(idx, size=n_zone, replace=False))

        if not sampled_land_indices:
            del glc_arr
            gc.collect()
            continue

        sampled_idx_unr = np.concatenate(sampled_land_indices)
        sampled_r_unr   = land_rows[sampled_idx_unr]
        sampled_c_unr   = land_cols_idx[sampled_idx_unr]

        not_event_unr = uglc_arr[sampled_r_unr, sampled_c_unr] != 1
        sampled_r_unr = sampled_r_unr[not_event_unr]
        sampled_c_unr = sampled_c_unr[not_event_unr]

        if len(sampled_r_unr) == 0:
            hb.log(f'  WARNING: No unrestricted controls sampled for year {year}')
            sampled_r_unr = np.array([], dtype=np.int32)
            sampled_c_unr = np.array([], dtype=np.int32)
        
        # RESTRICTED CONTROLS: sample from populated pixels using fast candidate batches
        hb.log(f'  Sampling restricted controls (population > 0 pixels) by zone...')
        sampled_land_indices_res = []
        n_zone_pixels = {zone: len(idx) for zone, idx in zone_idx.items()}
        n_total_zone_pixels = sum(n_zone_pixels.values())

        max_candidates_per_zone = 200000
        min_candidates_per_zone = 5000

        for zone, idx in zone_idx.items():
            if len(idx) == 0:
                continue

            n_zone_desired = max(1, round(n_controls_this_year * len(idx) / n_total_zone_pixels))

            # Remove known event pixels first so we only sample controls.
            zone_rows_all = land_rows[idx]
            zone_cols_all = land_cols_idx[idx]
            zone_non_event_mask = uglc_arr[zone_rows_all, zone_cols_all] != 1
            zone_pool = idx[zone_non_event_mask]
            if len(zone_pool) == 0:
                continue

            # First pass: sample a moderate candidate batch and keep populated points.
            cand_n_1 = min(len(zone_pool), max(min_candidates_per_zone, n_zone_desired * 10), max_candidates_per_zone)
            cand_1 = rng.choice(zone_pool, size=cand_n_1, replace=False)
            pop_cand_1 = _sample_raster_values(
                pop_path_local, land_rows[cand_1], land_cols_idx[cand_1], expected_shape=ref_shape, chunk=200000)
            good_1 = cand_1[pop_cand_1 > 0]

            if len(good_1) >= n_zone_desired:
                chosen = rng.choice(good_1, size=n_zone_desired, replace=False)
                sampled_land_indices_res.append(chosen)
                continue

            # Second pass only if needed, drawn from remaining pool.
            need = n_zone_desired - len(good_1)
            if cand_n_1 < len(zone_pool) and need > 0:
                remaining_pool = np.setdiff1d(zone_pool, cand_1, assume_unique=False)
                if len(remaining_pool) > 0:
                    cand_n_2 = min(len(remaining_pool), max(min_candidates_per_zone, need * 15), max_candidates_per_zone)
                    cand_2 = rng.choice(remaining_pool, size=cand_n_2, replace=False)
                    pop_cand_2 = _sample_raster_values(
                        pop_path_local, land_rows[cand_2], land_cols_idx[cand_2], expected_shape=ref_shape, chunk=200000)
                    good_2 = cand_2[pop_cand_2 > 0]
                    good_all = np.concatenate([good_1, good_2]) if len(good_1) > 0 else good_2
                else:
                    good_all = good_1
            else:
                good_all = good_1

            if len(good_all) > 0:
                draw_n = min(n_zone_desired, len(good_all))
                chosen = rng.choice(good_all, size=draw_n, replace=False)
                sampled_land_indices_res.append(chosen)

        if sampled_land_indices_res:
            sampled_idx_res = np.concatenate(sampled_land_indices_res)
            sampled_r_res = land_rows[sampled_idx_res]
            sampled_c_res = land_cols_idx[sampled_idx_res]
            hb.log(f'    Requested restricted controls: {n_controls_this_year}, realized: {len(sampled_r_res)}')
        else:
            sampled_r_res = np.array([], dtype=np.int32)
            sampled_c_res = np.array([], dtype=np.int32)

        # Process and save unrestricted controls
        if len(sampled_r_unr) > 0:
            n_ctrl_unr = len(sampled_r_unr)
            rain_vals_unr = _sample_raster_values(
                rain_path_local, sampled_r_unr, sampled_c_unr, expected_shape=ref_shape, chunk=200000)
            pop_vals_unr = _sample_raster_values(
                pop_path_local, sampled_r_unr, sampled_c_unr, expected_shape=ref_shape, chunk=200000)
            if has_sdr:
                sdr_obs_vals_unr = _sample_raster_values(
                    sdr_obs_path_local, sampled_r_unr, sampled_c_unr, expected_shape=ref_shape, chunk=200000)
                sdr_cf_vals_unr = _sample_raster_values(
                    sdr_cf_path_local, sampled_r_unr, sampled_c_unr, expected_shape=ref_shape, chunk=200000)
            else:
                sdr_obs_vals_unr = np.full(n_ctrl_unr, np.nan, dtype=np.float32)
                sdr_cf_vals_unr  = np.full(n_ctrl_unr, np.nan, dtype=np.float32)
            lulc_vals_unr = {
                var: _sample_raster_values(
                    lulc_paths_local[var], sampled_r_unr, sampled_c_unr, expected_shape=ref_shape, chunk=200000)
                for var in lulc_vars
            }
            refined_lulc_vals_unr = {
                var: _sample_raster_values(
                    refined_lulc_paths_local[var], sampled_r_unr, sampled_c_unr, expected_shape=ref_shape, chunk=200000)
                for var in refined_lulc_vars
            }
            deforestation_vals_unr = {
                col: _sample_raster_values(
                    deforestation_paths_local[col], sampled_r_unr, sampled_c_unr, expected_shape=ref_shape, chunk=200000)
                for col in deforestation_paths_local
            }
            slope_vals_unr = slope_arr[sampled_r_unr,  sampled_c_unr]
            roughness_vals_unr = (roughness_arr[sampled_r_unr, sampled_c_unr] if roughness_arr is not None
                                  else np.full(n_ctrl_unr, np.nan, dtype=np.float32))
            tpi_vals_unr = (tpi_arr[sampled_r_unr, sampled_c_unr] if tpi_arr is not None
                            else np.full(n_ctrl_unr, np.nan, dtype=np.float32))
            elev_stdev_vals_unr = (elev_stdev_arr[sampled_r_unr, sampled_c_unr] if elev_stdev_arr is not None
                                   else np.full(n_ctrl_unr, np.nan, dtype=np.float32))
            # ls_vals_unr    = ls_arr[sampled_r_unr,     sampled_c_unr]
            # twi_vals_unr   = twi_arr[sampled_r_unr,    sampled_c_unr]
            gaez_vals_unr  = gaez_arr[sampled_r_unr,   sampled_c_unr]
            roads_vals_unr = (roads_arr[sampled_r_unr,  sampled_c_unr]
                            if roads_arr  is not None
                            else np.full(n_ctrl_unr, np.nan, dtype=np.float32))
            faults_vals_unr = (faults_arr[sampled_r_unr, sampled_c_unr]
                            if faults_arr is not None
                            else np.full(n_ctrl_unr, np.nan, dtype=np.float32))
            motorized_travel_vals_unr = (motorized_travel_arr[sampled_r_unr, sampled_c_unr]
                                         if motorized_travel_arr is not None
                                         else np.full(n_ctrl_unr, np.nan, dtype=np.float32))
            walking_travel_vals_unr   = (walking_travel_arr[sampled_r_unr, sampled_c_unr]
                                         if walking_travel_arr is not None
                                         else np.full(n_ctrl_unr, np.nan, dtype=np.float32))

            control_batch_unr = pd.DataFrame({
                'year':                np.full(n_ctrl_unr, year, dtype=np.int16),
                'row':                 sampled_r_unr.astype(np.int32),
                'col':                 sampled_c_unr.astype(np.int32),
                'landslide':           np.zeros(n_ctrl_unr, dtype=np.int8),
                'slope_degree':        slope_vals_unr,
                'roughness':           roughness_vals_unr,
                'tpi':                 tpi_vals_unr,
                'elev_stdev':          elev_stdev_vals_unr,
                # 'ls_factor':           ls_vals_unr,
                # 'twi':                 twi_vals_unr,
                'road_density':        roads_vals_unr,
                'dist_to_fault_km':    faults_vals_unr,
                'motorized_travel_time': motorized_travel_vals_unr,
                'walking_travel_time':   walking_travel_vals_unr,
                'rain_max_daily':      rain_vals_unr,
                'population':          pop_vals_unr,
                'is_populated':        (pop_vals_unr > 0),
                'mortality':           np.zeros(n_ctrl_unr, dtype=np.float32),
                'avoided_erosion_obs': sdr_obs_vals_unr,
                'avoided_erosion_cf':  sdr_cf_vals_unr,
                'gaez_zone':           gaez_vals_unr.astype(np.int16),
                'control_source':      'unrestricted',
            })
            for var in lulc_vars:
                control_batch_unr[f'{var}_share'] = lulc_vals_unr[var]
            for var in refined_lulc_vars:
                control_batch_unr[f'{var}_share_refined'] = refined_lulc_vals_unr[var]
            for col in deforestation_vals_unr:
                control_batch_unr[col] = deforestation_vals_unr[col]
            
            year_controls_unr_path = os.path.join(intermediate_tables_dir, f'year_{year}_controls_unrestricted.parquet')
            control_batch_unr.to_parquet(year_controls_unr_path, index=False)
            hb.log(f'    Saved {len(control_batch_unr)} unrestricted controls to {os.path.basename(year_controls_unr_path)}')

        # Process and save restricted controls
        if len(sampled_r_res) > 0:
            n_ctrl_res = len(sampled_r_res)
            rain_vals_res = _sample_raster_values(
                rain_path_local, sampled_r_res, sampled_c_res, expected_shape=ref_shape, chunk=200000)
            pop_vals_res = _sample_raster_values(
                pop_path_local, sampled_r_res, sampled_c_res, expected_shape=ref_shape, chunk=200000)
            if has_sdr:
                sdr_obs_vals_res = _sample_raster_values(
                    sdr_obs_path_local, sampled_r_res, sampled_c_res, expected_shape=ref_shape, chunk=200000)
                sdr_cf_vals_res = _sample_raster_values(
                    sdr_cf_path_local, sampled_r_res, sampled_c_res, expected_shape=ref_shape, chunk=200000)
            else:
                sdr_obs_vals_res = np.full(n_ctrl_res, np.nan, dtype=np.float32)
                sdr_cf_vals_res  = np.full(n_ctrl_res, np.nan, dtype=np.float32)
            lulc_vals_res = {
                var: _sample_raster_values(
                    lulc_paths_local[var], sampled_r_res, sampled_c_res, expected_shape=ref_shape, chunk=200000)
                for var in lulc_vars
            }
            refined_lulc_vals_res = {
                var: _sample_raster_values(
                    refined_lulc_paths_local[var], sampled_r_res, sampled_c_res, expected_shape=ref_shape, chunk=200000)
                for var in refined_lulc_vars
            }
            deforestation_vals_res = {
                col: _sample_raster_values(
                    deforestation_paths_local[col], sampled_r_res, sampled_c_res, expected_shape=ref_shape, chunk=200000)
                for col in deforestation_paths_local
            }
            slope_vals_res = slope_arr[sampled_r_res,  sampled_c_res]
            roughness_vals_res = (roughness_arr[sampled_r_res, sampled_c_res] if roughness_arr is not None
                                  else np.full(n_ctrl_res, np.nan, dtype=np.float32))
            tpi_vals_res = (tpi_arr[sampled_r_res, sampled_c_res] if tpi_arr is not None
                            else np.full(n_ctrl_res, np.nan, dtype=np.float32))
            elev_stdev_vals_res = (elev_stdev_arr[sampled_r_res, sampled_c_res] if elev_stdev_arr is not None
                                   else np.full(n_ctrl_res, np.nan, dtype=np.float32))
            # ls_vals_res    = ls_arr[sampled_r_res,     sampled_c_res]
            # twi_vals_res   = twi_arr[sampled_r_res,    sampled_c_res]
            gaez_vals_res  = gaez_arr[sampled_r_res,   sampled_c_res]
            roads_vals_res = (roads_arr[sampled_r_res,  sampled_c_res]
                            if roads_arr  is not None
                            else np.full(n_ctrl_res, np.nan, dtype=np.float32))
            faults_vals_res = (faults_arr[sampled_r_res, sampled_c_res]
                            if faults_arr is not None
                            else np.full(n_ctrl_res, np.nan, dtype=np.float32))
            motorized_travel_vals_res = (motorized_travel_arr[sampled_r_res, sampled_c_res]
                                         if motorized_travel_arr is not None
                                         else np.full(n_ctrl_res, np.nan, dtype=np.float32))
            walking_travel_vals_res   = (walking_travel_arr[sampled_r_res, sampled_c_res]
                                         if walking_travel_arr is not None
                                         else np.full(n_ctrl_res, np.nan, dtype=np.float32))

            control_batch_res = pd.DataFrame({
                'year':                np.full(n_ctrl_res, year, dtype=np.int16),
                'row':                 sampled_r_res.astype(np.int32),
                'col':                 sampled_c_res.astype(np.int32),
                'landslide':           np.zeros(n_ctrl_res, dtype=np.int8),
                'slope_degree':        slope_vals_res,
                'roughness':           roughness_vals_res,
                'tpi':                 tpi_vals_res,
                'elev_stdev':          elev_stdev_vals_res,
                # 'ls_factor':           ls_vals_res,
                # 'twi':                 twi_vals_res,
                'road_density':        roads_vals_res,
                'dist_to_fault_km':    faults_vals_res,
                'motorized_travel_time': motorized_travel_vals_res,
                'walking_travel_time':   walking_travel_vals_res,
                'rain_max_daily':      rain_vals_res,
                'population':          pop_vals_res,
                'is_populated':        (pop_vals_res > 0),
                'mortality':           np.zeros(n_ctrl_res, dtype=np.float32),
                'avoided_erosion_obs': sdr_obs_vals_res,
                'avoided_erosion_cf':  sdr_cf_vals_res,
                'gaez_zone':           gaez_vals_res.astype(np.int16),
                'control_source':      'restricted',
            })
            for var in lulc_vars:
                control_batch_res[f'{var}_share'] = lulc_vals_res[var]
            for var in refined_lulc_vars:
                control_batch_res[f'{var}_share_refined'] = refined_lulc_vals_res[var]
            for col in deforestation_vals_res:
                control_batch_res[col] = deforestation_vals_res[col]
            
            year_controls_res_path = os.path.join(intermediate_tables_dir, f'year_{year}_controls_restricted.parquet')
            control_batch_res.to_parquet(year_controls_res_path, index=False)
            hb.log(f'    Saved {len(control_batch_res)} restricted controls to {os.path.basename(year_controls_res_path)}')
        
        # Force garbage collection to free annual arrays.
        del uglc_arr
        gc.collect()
    # ------------------------------------------------------------------
    hb.log('Loading all year-specific results...')
    
    # Load all event and control parquet files
    all_events = []
    all_controls = []
    
    for year in estimation_years:
        year_events_path = os.path.join(intermediate_tables_dir, f'year_{year}_events.parquet')
        year_controls_unr_path = os.path.join(intermediate_tables_dir, f'year_{year}_controls_unrestricted.parquet')
        year_controls_res_path = os.path.join(intermediate_tables_dir, f'year_{year}_controls_restricted.parquet')
        
        if os.path.exists(year_events_path):
            df_ev = pd.read_parquet(year_events_path)
            all_events.append(df_ev)
            hb.log(f'  Loaded {len(df_ev)} events from year {year}')
        
        if os.path.exists(year_controls_unr_path):
            df_ctrl_unr = pd.read_parquet(year_controls_unr_path)
            all_controls.append(df_ctrl_unr)
            hb.log(f'  Loaded {len(df_ctrl_unr)} unrestricted controls from year {year}')
        
        if os.path.exists(year_controls_res_path):
            df_ctrl_res = pd.read_parquet(year_controls_res_path)
            all_controls.append(df_ctrl_res)
            hb.log(f'  Loaded {len(df_ctrl_res)} restricted controls from year {year}')
    
    hb.log('Combining all records...')
    
    if all_events:
        df_events = pd.concat(all_events, ignore_index=True)
    else:
        df_events = pd.DataFrame()
    
    if all_controls:
        df_controls = pd.concat(all_controls, ignore_index=True)
    else:
        df_controls = pd.DataFrame()

    # Warn if overall fraction of sampled controls that are populated is low
    try:
        if not df_controls.empty:
            frac_controls_pop = float((df_controls['population'] > 0).sum()) / float(len(df_controls))
            hb.log(f'Overall controls populated fraction: {frac_controls_pop:.3%}')
            if frac_controls_pop < float(min_pop_control_frac):
                hb.log(f'WARNING: Overall sampled controls only {frac_controls_pop:.3%} populated (< {min_pop_control_frac:.1%}). Consider reviewing control sampling strategy.')
    except Exception:
        hb.log('WARNING: could not compute overall populated fraction for controls')
    
    # Add control_source to events (for consistency with Option B structure)
    if not df_events.empty:
        df_events['control_source'] = 'event'
    
    df = pd.concat([df_events, df_controls], ignore_index=True)

    # Core covariates required; roads/faults/SDR allowed to be NaN
    required_covariates = (
        ['slope_degree', 'roughness', 'tpi', 'elev_stdev', 'rain_max_daily', 'population']
        + [f'{v}_share' for v in lulc_vars]
        + [col for col, _folder, _source, _window in deforestation_specs]
    )
    n_before = len(df)
    df = df.dropna(subset=required_covariates)
    hb.log(f'Dropped {n_before - len(df):,} rows with missing required '
           f'covariates ({len(df):,} rows remain)')

    # Coverage summary for optional columns
    for col in ['road_density', 'dist_to_fault_km',
                'avoided_erosion_obs', 'avoided_erosion_cf']:
        n_valid = df[col].notna().sum()
        hb.log(f'  {col}: {n_valid:,} / {len(df):,} valid '
               f'({100 * n_valid / len(df):.1f}%)')

    unexpected = df.loc[~df['gaez_zone'].between(1, 57), 'gaez_zone'].unique()
    if len(unexpected) > 0:
        hb.log(f'WARNING: unexpected gaez_zone values: '
               f'{sorted(unexpected.tolist())}')

    # ---- Dtypes ------------------------------------------------------
    dtype_map = {
        'year':                np.int16,
        'row':                 np.int32,
        'col':                 np.int32,
        'landslide':           np.int8,
        'slope_degree':        np.float32,
        'roughness':           np.float32,
        'tpi':                 np.float32,
        'elev_stdev':          np.float32,
        'road_density':        np.float32,
        'dist_to_fault_km':    np.float32,
        'motorized_travel_time': np.float32,
        'walking_travel_time':   np.float32,
        'rain_max_daily':      np.float32,
        'population':          np.float32,
        'is_populated':        np.bool_,
        'control_source':      'object',
        'mortality':           np.float32,
        'avoided_erosion_obs': np.float32,
        'avoided_erosion_cf':  np.float32,
        'gaez_zone':           np.int16,
    }
    for v in lulc_vars:
        dtype_map[f'{v}_share'] = np.float32
    for v in refined_lulc_vars:
        dtype_map[f'{v}_share_refined'] = np.float32
    for col, _folder, _source, _window in deforestation_specs:
        dtype_map[col] = np.float32
    df = df.astype(dtype_map)

    df.to_parquet(out_parquet, index=False)
    hb.log(f'Saved: {out_parquet}  ({len(df):,} rows, '
           f'{os.path.getsize(out_parquet) / 1e6:.1f} MB)')

    # ------------------------------------------------------------------
    # 5. Sampling metadata for intercept correction
    # ------------------------------------------------------------------
    n_controls_sampled   = len(df_controls)
    sampling_fraction    = (n_controls_sampled / n_total_non_event_pixel_years
                            if n_total_non_event_pixel_years > 0 else np.nan)
    intercept_correction = (float(np.log(1.0 / sampling_fraction))
                            if np.isfinite(sampling_fraction)
                            and sampling_fraction > 0 else np.nan)

    pd.DataFrame([{
        'event_source':                 event_source,
        'estimation_start_year':        estimation_years[0],
        'estimation_end_year':          estimation_years[-1],
        'estimation_n_years':           len(estimation_years),
        'n_events':                      len(df_events),
        'n_controls_sampled':            n_controls_sampled,
        'n_total_non_event_pixel_years': n_total_non_event_pixel_years,
        'sampling_fraction':             sampling_fraction,
        'control_ratio':                 control_ratio,
        'intercept_correction':          intercept_correction,
    }]).to_csv(out_meta, index=False)

    hb.log(f'Sampling fraction:             {sampling_fraction:.4e}')
    hb.log(f'Intercept correction log(1/f): +{intercept_correction:.4f}')
    hb.log('IMPORTANT: intercept_correction will be applied to hazard predictions')
    hb.log('  so absolute probability levels reflect case-control sampling.')
    hb.log(f'Saved metadata: {out_meta}')

    return p


def _compute_logit_marginal_effects(hazard_result, X_hazard, scaler_info, hazard_predictors_final,
                                    gaez_ref_zone, intercept_correction=0.0):
    """
    Compute marginal effects for logit model at the mean and representative scenarios.
    
    Returns dict with:
      - 'at_mean': marginal effects at mean covariate values
      - 'scenarios': dict of scenario-based marginal effects
    """
    import numpy as np
    
    # Apply intercept correction for case-control sampling.
    params = hazard_result.params.copy()
    if np.isfinite(intercept_correction) and intercept_correction != 0:
        params.loc['const'] = float(params.get('const', 0.0) + intercept_correction)

    # Compute linear predictor at the mean
    X_mean = X_hazard.mean(axis=0)
    eta_mean = float(np.dot(X_mean, params))
    
    # Logit inverse link: P = 1 / (1 + exp(-η))
    eta_mean_clip = np.clip(eta_mean, -500, 500)
    p_mean = 1.0 / (1.0 + np.exp(-eta_mean_clip))
    
    # Derivative factor for logit
    me_factor = p_mean * (1.0 - p_mean)
    
    # Marginal effects at the mean (only for main predictors, not GAEZ dummies)
    me_at_mean = {}
    for term in params.index:
        if term == 'const' or term.startswith('gaez_') or term.startswith('year_'):
            continue
        beta = float(params[term])
        # Check if this predictor is standardized
        if term in scaler_info:
            # For standardized variables, ME is per SD unit; convert to per unit of original scale
            sigma = scaler_info[term].get('std', 1.0)
            if sigma > 0:
                me_at_mean[term] = float(me_factor * beta / sigma)
            else:
                me_at_mean[term] = 0.0
        else:
            # For non-standardized (e.g., GAEZ dummies), ME is as-is
            me_at_mean[term] = float(me_factor * beta)
    
    # Compute marginal effects at representative scenarios for key predictors
    scenarios = {}
    key_scenarios = {
        'slope_degree': [5.0, 15.0, 30.0],  # Low, medium, high slope
        'deforestation_forest_share_1yr': [0.0, 0.05, 0.15],  # No loss, moderate, high loss
    }
    
    for var_name, var_values in key_scenarios.items():
        if var_name not in scaler_info:
            continue
        
        scenarios[var_name] = {}
        mu = scaler_info[var_name]['mean']
        sigma = scaler_info[var_name]['std']
        
        for val in var_values:
            # Compute predictor at this value, holding others at mean
            X_scenario = X_mean.copy()
            X_scenario[var_name] = (val - mu) / sigma if sigma > 0 else 0
            
            eta_scenario = float(np.dot(X_scenario, params))
            eta_scenario_clip = np.clip(eta_scenario, -500, 500)
            p_scenario = 1.0 / (1.0 + np.exp(-eta_scenario_clip))
            
            # Store predicted probability at this scenario
            label = f'P({var_name}={val:.2f})'
            scenarios[var_name][label] = float(p_scenario)
    
    return {
        'at_mean': me_at_mean,
        'p_at_mean': float(p_mean),
        'scenarios': scenarios,
    }


def estimate_hazard_model(p):
    """
    Fit two separate models for forest value counterfactual analysis:
    
    1. Hazard Model (Stage 1): Logit(landslide) ~ terrain + infrastructure + deforestation + FE
       Predicts: p(landslide | covariates), used in scenario comparison
       
    2. Severity Model: Tweedie(mortality | landslide=1) ~ population + rain + terrain + FE
       Predicts: E[mortality | observed covariates, landslide occurred]
       Used to convert hazard probability changes into mortality impacts

    Output files
    ----------------------------------
    hazard_model.json                    — Logit model coefficients + scaler
    hazard_model_summary.txt             — Full model summary
    severity_model.json                  — Tweedie model coefficients + scaler
    severity_model_summary.txt           — Full model summary
    """
    if not p.run_this:
        return p

    import statsmodels.api as sm
    import gc

    parquet_path = os.path.join(p.build_estimation_table_dir,
                                'estimation_table.parquet')
    meta_path    = os.path.join(p.build_estimation_table_dir,
                                'sampling_meta.csv')

    if not os.path.exists(parquet_path):
        hb.log(f'Estimation table not found at {parquet_path}, skipping.')
        return p
    
    mort_summary_path = os.path.join(p.cur_dir, 'mortality_model_summary.txt')
    if os.path.exists(mort_summary_path):
        hb.log(f'Mortality model summary already exists at {mort_summary_path}, skipping estimation.')
        return p

    sampling_meta = None
    intercept_correction = 0.0
    if os.path.exists(meta_path):
        sampling_meta = pd.read_csv(meta_path).iloc[0].to_dict()
        intercept_correction = float(sampling_meta.get('intercept_correction', 0.0) or 0.0)
        if not np.isfinite(intercept_correction):
            hb.log('  WARNING: invalid intercept_correction in sampling_meta.csv; using 0.0')
            intercept_correction = 0.0
    else:
        hb.log(f'  WARNING: sampling metadata not found at {meta_path}; hazard intercept correction disabled.')

    # ------------------------------------------------------------------
    # 1. Load table
    # ------------------------------------------------------------------
    hb.log('Loading estimation table...')
    df = pd.read_parquet(parquet_path)
    hb.log(
        f'  {len(df):,} rows  |  '
        f'landslide events: {int(df["landslide"].sum()):,}  |  '
        f'pixels with mortality > 0: {int((df["mortality"] > 0).sum()):,}'
    )
    hb.log(f'  Mortality stats — mean: {df["mortality"].mean():.4f}  '
           f'max: {df["mortality"].max():.2f}  '
           f'zeros: {(df["mortality"] == 0).mean():.1%}')
    
    # Diagnostic: show which columns exist and their NaN counts
    hb.log('  Column info before cleaning:')
    for col in ['landslide', 'mortality', 'population', 'rain_max_daily',
                'slope_degree', 'roughness', 'tpi', 'elev_stdev', 'road_density', 'dist_to_fault_km',
                'deforestation_forest_share_1yr', 'deforestation_forest_share_3yr',
                'forest_share', 'othernat_share', 'gaez_zone']:
        if col in df.columns:
            n_nan = df[col].isna().sum()
            n_inf = np.isinf(df[col]).sum() if df[col].dtype in [np.float32, np.float64] else 0
            hb.log(f'    {col}: {n_nan:,} NaN, {n_inf:,} inf, {len(df) - n_nan - n_inf:,} valid')
        else:
            hb.log(f'    {col}: MISSING FROM DATAFRAME')

    # ------------------------------------------------------------------
    # 2. Drop rows with missing required covariates and infinite values
    # ------------------------------------------------------------------
    required = ['landslide', 'mortality', 'population', 'rain_max_daily',
                'slope_degree', 'roughness', 'tpi', 'elev_stdev', 'road_density', 'dist_to_fault_km',
                'deforestation_forest_share_1yr', 'deforestation_forest_share_3yr',
                'forest_share', 'othernat_share', 'gaez_zone']
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f'Missing columns in estimation table: {missing_cols}')

    n_before = len(df)
    
    # Drop NaN values
    hb.log('  Dropping rows with NaN in required columns...')
    n_with_nans = df[required].isna().any(axis=1).sum()
    hb.log(f'    {n_with_nans:,} rows have NaN in required columns')
    df = df.dropna(subset=required).copy()
    hb.log(f'    After dropna: {len(df):,} rows remain')
    
    # Drop infinite values in continuous columns
    continuous_cols = ['population', 'rain_max_daily', 'slope_degree', 'roughness', 'tpi', 'elev_stdev',
                      'road_density', 'dist_to_fault_km', 'forest_share', 'othernat_share',
                      'deforestation_forest_share_1yr', 'deforestation_forest_share_3yr',
                      'motorized_travel_time', 'walking_travel_time']
    for col in continuous_cols:
        if col in df.columns:
            n_inf = np.isinf(df[col]).sum()
            if n_inf > 0:
                hb.log(f'    Removing {n_inf:,} rows with inf in {col}')
                df = df[~np.isinf(df[col])]
            
            # Drop travel-time NoData values (-9999.0)
            if col in ['motorized_travel_time', 'walking_travel_time']:
                n_nodata = (df[col] == -9999.0).sum()
                if n_nodata > 0:
                    hb.log(f'    Removing {n_nodata:,} rows with NoData (-9999) in {col}')
                    df = df[df[col] != -9999.0]
    
    # Drop infinite values in outcome
    df = df[~np.isinf(df['landslide'])]
    df = df[~np.isinf(df['mortality'])]
    
    
    hb.log(f'Dropped {n_before - len(df):,} rows with NaN or inf values '
           f'({len(df):,} remain)')

    # Drop implausible GAEZ zones
    df = df[df['gaez_zone'].between(1, 57)].copy()

    # Aggressive coarsening: map original GAEZ zones (1..57) into a small
    # number of AEZ groups to stabilise fixed-effects.
    hb.log('  Mapping GAEZ zones into coarse AEZ groups for FE...')
    # Map using lookup array defined at module top
    df['gaez_zone'] = df['gaez_zone'].apply(_zone_to_group).astype(np.int16)

    # Remove any rows mapped to 0 (invalid/unmapped)
    n_before_map = len(df)
    df = df[df['gaez_zone'] > 0].copy()
    hb.log(f'    Removed {n_before_map - len(df):,} rows with invalid GAEZ after grouping')

    # Compute group counts and drop extremely small groups (<10 rows)
    group_counts = df['gaez_zone'].value_counts().sort_index()
    small_groups = group_counts[group_counts < 10].index.tolist()
    if small_groups:
        hb.log(f'    Dropping very small AEZ groups: {small_groups}')
        df = df[~df['gaez_zone'].isin(small_groups)].copy()

    hb.log(f'  {len(df):,} rows after AEZ grouping/filtering')

    # ------------------------------------------------------------------
    # 3. Winsorize continuous predictors at 99.9th percentile
    # ------------------------------------------------------------------
    winsor_cols = ['population', 'rain_max_daily', 'slope_degree', 'roughness', 'tpi', 'elev_stdev',
                   'road_density', 'dist_to_fault_km', 'motorized_travel_time', 'walking_travel_time']
    winsor_cols += [
        'deforestation_forest_share_1yr', 'deforestation_forest_share_3yr',
    ]
    scaler_info = {}
    winsor_caps = {}
    for col in winsor_cols:
        cap = float(df[col].quantile(0.999))
        winsor_caps[col] = cap
        n_clipped = int((df[col] > cap).sum())
        hb.log(f'  Winsorizing {col} at 99.9th pct ({cap:.4f}): '
               f'{n_clipped:,} clipped')
        df[col] = df[col].clip(upper=cap)

    # Keep a copy of the winsorized but unstandardized data for severity model fitting
    # CRITICAL: severity model must compute scaler on raw (un-standardized) data
    df_raw = df.copy()  # Keep FULL dataframe before hazard standardization
    population_stage2_raw = df['population'].astype(np.float64).copy()

    # ------------------------------------------------------------------
    # 4. Standardize continuous predictors (mean 0, SD 1) — FOR HAZARD MODEL ONLY
    # ------------------------------------------------------------------
    for col in winsor_cols:
        # Convert to float64 FIRST to avoid overflow in mean/std computation
        df[col] = df[col].astype(np.float64)
        mu    = float(df[col].mean())
        sigma = float(df[col].std())
        if sigma == 0:
            hb.log(f'  WARNING: {col} has zero std — skipping standardization')
            scaler_info[col] = {'mean': mu, 'std': 1.0, 'cap': winsor_caps.get(col)}
        else:
            df[col] = (df[col] - mu) / sigma
            scaler_info[col] = {'mean': mu, 'std': sigma, 'cap': winsor_caps.get(col)}
    
    # Validate after standardization — STRICT: fail on any NaN/inf
    hb.log('Validating standardization...')
    for col in winsor_cols:
        n_nan_after = df[col].isna().sum()
        n_inf_after = np.isinf(df[col]).sum()
        if n_nan_after > 0 or n_inf_after > 0:
            error_msg = (f'Standardization failed for {col}: {n_nan_after} NaN and {n_inf_after} inf. '
                         f'Data integrity compromised. Aborting.')
            hb.log(f'ERROR: {error_msg}')
            raise ValueError(error_msg)
        else:
            hb.log(f'  {col}: OK')
    
    hb.log('Standardization complete — all columns valid.')
    
    # Also standardize vegetation shares (center to training mean)
    for col in ['forest_share', 'othernat_share']:
        mu = float(df[col].mean())
        scaler_info[col] = {'mean': mu, 'std': 1.0}
        df[col] = df[col] - mu  # Keep as float64


    # ------------------------------------------------------------------
    # 5. GAEZ fixed-effect dummies
    # ------------------------------------------------------------------
    # Choose preferred reference group: prefer groups containing historically
    # chosen zones, otherwise take the most populous group.
    preferred_original_zones = [50, 49, 42, 41, 40, 33]
    preferred_groups = [g for g in (_zone_to_group(z) for z in preferred_original_zones) if g > 0]
    preferred_groups = [g for g in preferred_groups if g in df['gaez_zone'].values]
    ref_zone = next((g for g in preferred_groups), int(df['gaez_zone'].value_counts().idxmax()))
    all_zones     = sorted(df['gaez_zone'].unique().tolist())
    non_ref_zones = [z for z in all_zones if z != ref_zone]
    hb.log(f'GAEZ reference zone: {ref_zone}  |  dummies: {len(non_ref_zones)}')

    gaez_cols = []
    new_cols  = {}
    for z in non_ref_zones:
        col = f'gaez_{z}'
        new_cols[col] = (df['gaez_zone'] == z).astype(np.float32)
        gaez_cols.append(col)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # ------------------------------------------------------------------
    # Year fixed effects: create year dummies with a reference year
    # ------------------------------------------------------------------
    hb.log('  Creating year fixed-effects dummies...')
    years_present = sorted(df['year'].unique().tolist())
    # Choose the most frequent year as reference to improve stability
    ref_year = int(df['year'].value_counts().idxmax())
    year_cols = []
    for y in years_present:
        if int(y) == ref_year:
            continue
        col = f'year_{int(y)}'
        df[col] = (df['year'] == int(y)).astype(np.float32)
        year_cols.append(col)
    hb.log(f'    Year FE reference: {ref_year}  |  dummies: {len(year_cols)}')

    # ------------------------------------------------------------------
    # 6. HAZARD MODEL: Fit Logit for landslide probability
    # ------------------------------------------------------------------
    # Hazard predictors: terrain + infrastructure + deforestation
    # Used to predict p(landslide) under observed and counterfactual scenarios
    hazard_predictors = [
        'slope_degree', 'roughness', 'tpi', 'elev_stdev', 'road_density', 'dist_to_fault_km',
        'deforestation_forest_share_1yr', 'deforestation_forest_share_3yr', 'rain_max_daily'
    ]

    # Add AEZ fixed effects
    hazard_predictors += gaez_cols

    # Build dataset: all observations (events + all controls)
    df_hazard = df.copy()

    # Filter predictors to only varying ones
    hazard_predictors_final = []
    dropped_hazard_predictors = []
    for col in hazard_predictors:
        series = pd.to_numeric(df_hazard[col], errors='coerce')
        sigma = float(series.std())
        if not np.isfinite(sigma) or sigma <= 1e-12:
            dropped_hazard_predictors.append(col)
        else:
            hazard_predictors_final.append(col)

    if dropped_hazard_predictors:
        hb.log('  Dropping non-varying hazard predictors: ' + ', '.join(dropped_hazard_predictors))
    if not hazard_predictors_final:
        raise RuntimeError('No varying predictors remain for hazard model after filtering constants.')

    X_hazard = sm.add_constant(df_hazard[hazard_predictors_final].astype(np.float32))
    y_landslide = df_hazard['landslide'].astype(np.float32)

    hb.log('\n=== HAZARD MODEL: Logit(landslide) ===')
    hb.log(f'Fitting with {len(hazard_predictors_final)} predictors on {len(df_hazard)} observations...')

    try:
        hazard_model = sm.Logit(y_landslide, X_hazard)
        hazard_result = hazard_model.fit(maxiter=200, disp=False)
        hb.log(f'  Converged: {hazard_result.converged}')
    except Exception as e:
        raise RuntimeError(f'ERROR fitting hazard model: {type(e).__name__}: {e}')
    
    # Save hazard model
    hazard_summary_path = os.path.join(p.cur_dir, 'hazard_model_summary.txt')
    with open(hazard_summary_path, 'w') as f:
        f.write(str(hazard_result.summary()))
    hb.log(f'  Saved: {hazard_summary_path}')
    
    # Compute marginal effects using logit link
    hb.log('Computing logit marginal effects...')
    marginal_effects = _compute_logit_marginal_effects(
        hazard_result, X_hazard, scaler_info, hazard_predictors_final, ref_zone,
        intercept_correction=intercept_correction
    )
    hb.log(f'  P(landslide) at mean covariates: {marginal_effects["p_at_mean"]:.4f}')
    hb.log(f'  Computed marginal effects for {len(marginal_effects["at_mean"])} predictors')
    
    hazard_json_path = os.path.join(p.cur_dir, 'hazard_model.json')
    hazard_output = {
        'model_key':       'hazard_logit_fe',
        'family':          'logit',
        'link':            'logit',
        'dep_var':         'landslide',
        'converged':       bool(hazard_result.converged),
        'n_obs':           int(len(y_landslide)),
        'n_events':        int(y_landslide.sum()),
        'dep_mean':        float(y_landslide.mean()),
        'llf':             float(hazard_result.llf) if np.isfinite(hazard_result.llf) else None,
        'll_null':         float(getattr(hazard_result, 'llnull', np.nan)) if np.isfinite(getattr(hazard_result, 'llnull', np.nan)) else None,
        'aic':             float(hazard_result.aic) if hasattr(hazard_result, 'aic') else None,
        'pseudo_r2_mcfadden': (
            float(1.0 - (hazard_result.llf / hazard_result.llnull))
            if hasattr(hazard_result, 'llnull') and np.isfinite(hazard_result.llf) and np.isfinite(hazard_result.llnull) and hazard_result.llnull != 0
            else None
        ),
        'predictors':      hazard_predictors_final,
        'predictors_dropped': dropped_hazard_predictors,
        'gaez_ref_zone':   ref_zone,
        'intercept_correction': intercept_correction,
        'scaler':          scaler_info,
        'coefficients':    hazard_result.params.to_dict(),
        'coefficients_corrected': {
            **hazard_result.params.to_dict(),
            'const': float(hazard_result.params.get('const', 0.0) + intercept_correction),
        },
        'std_errors':      hazard_result.bse.to_dict(),
        'pvalues':         hazard_result.pvalues.to_dict(),
        'marginal_effects': marginal_effects,
    }
    with open(hazard_json_path, 'w') as f:
        json.dump(hazard_output, f, indent=2)
    hb.log(f'  Saved: {hazard_json_path}')
    

    # ------------------------------------------------------------------
    # 7. SEVERITY MODEL: Predict E[mortality | landslide=1]
    # ------------------------------------------------------------------
    hb.log('\n=== SEVERITY MODEL (E[mortality | landslide=1]) ===')
    # Use df_raw (un-standardized) so scaler computes on actual data distribution
    df_sev = df_raw[(df_raw['landslide'] == 1) & (df_raw['mortality'] >= 0) & (df_raw['population'] > 0)].copy()
    n_sev = len(df_sev)
    n_sev_pos = (df_sev['mortality'] > 0).sum()
    hb.log(f'Fitting on {n_sev:,} landslide-positive pixels ({n_sev_pos:,} with mortality > 0)')
    
    df_sev['population_log1p'] = np.log1p(df_sev['population'].clip(lower=0))
    sev_cont_preds = ['population_log1p', 'rain_max_daily', 'slope_degree', 'road_density']
    sev_scaler_info = {}
    # Predictors are standardized (mean 0, SD 1) after winsorization.
    # No z-score cap: let the log link handle natural variation.
    # Note: raw data already winsorized at 99.9th percentile in estimation_table.
    
    for col in sev_cont_preds:
        mu = float(df_sev[col].mean())
        sigma = float(df_sev[col].std())
        if sigma > 0:
            df_sev[f'std_{col}'] = (df_sev[col] - mu) / sigma
            sev_scaler_info[col] = {'mean': mu, 'std': sigma}
        else:
            df_sev[f'std_{col}'] = 0.0
            sev_scaler_info[col] = {'mean': mu, 'std': 1.0}
    
    # GAEZ dummies
    sev_all_zones = sorted(df_sev['gaez_zone'].unique().tolist())
    sev_ref_zone = int(df_sev['gaez_zone'].value_counts().idxmax())
    sev_gaez_cols = []
    for z in sev_all_zones:
        if int(z) != sev_ref_zone:
            col = f'gaez_{int(z)}'
            df_sev[col] = (df_sev['gaez_zone'] == z).astype(np.float32)
            sev_gaez_cols.append(col)
    
    sev_predictors = [f'std_{c}' for c in sev_cont_preds] + sev_gaez_cols
    X_sev = sm.add_constant(df_sev[sev_predictors].astype(np.float32))
    y_sev = df_sev['mortality'].astype(np.float32)
    
    # ===== HURDLE MODEL: Two-stage approach for zero-inflated mortality =====
    hb.log('\n=== HURDLE MODEL (Two-stage) ===')
    
    # Stage 1: Hurdle (Logit) - P(mortality > 0 | landslide=1, X)
    # Drop problematic GAEZ zones (perfect separation, too few positive obs) to stabilize logit
    hb.log('  Stage 1: Fitting hurdle (logit) for P(mortality > 0)...')
    y_hurdle = (df_sev['mortality'] > 0).astype(np.float32)
    
    # Identify and remove problematic GAEZ zones from hurdle model
    hurdle_gaez_cols = []
    problematic_zones = []
    for col in sev_gaez_cols:
        zone_num = int(col.split('_')[1])
        zone_data = df_sev[df_sev['gaez_zone'] == zone_num]
        n_zone = len(zone_data)
        n_positive = (zone_data['mortality'] > 0).sum()
        n_zero = (zone_data['mortality'] == 0).sum()
        
        # Keep zone only if: n >= 10, n_positive >= 5, AND not perfect separation
        if n_zone >= 10 and n_positive >= 5 and n_positive < n_zone:
            hurdle_gaez_cols.append(col)
        else:
            problematic_zones.append((zone_num, n_zone, n_positive))
    
    if problematic_zones:
        hb.log(f'    Dropping {len(problematic_zones)} problematic GAEZ zones from hurdle logit:')
        for zone_num, n, n_pos in problematic_zones[:10]:  # Show first 10
            hb.log(f'      Zone {zone_num}: n={n}, n_positive={n_pos}')
        if len(problematic_zones) > 10:
            hb.log(f'      ... and {len(problematic_zones) - 10} more')
    
    hurdle_predictors = [f'std_{c}' for c in sev_cont_preds] + hurdle_gaez_cols
    X_hurdle = sm.add_constant(df_sev[hurdle_predictors].astype(np.float32))
    
    hurdle_model = sm.Logit(y_hurdle, X_hurdle).fit(maxiter=150, disp=False)
    hb.log(f'    Hurdle converged: {hurdle_model.converged}')
    hb.log(f'    Fraction positive: {float(y_hurdle.mean()):.4f}')
    
    # Stage 2: Severity (Tweedie GLM) - E[mortality | mortality > 0, X]
    hb.log('  Stage 2: Fitting severity (Tweedie GLM) for positive mortality only...')
    df_sev_pos = df_sev[df_sev['mortality'] > 0].copy()
    n_sev_pos = len(df_sev_pos)
    X_sev_pos = sm.add_constant(df_sev_pos[sev_predictors].astype(np.float32))
    y_sev_pos = df_sev_pos['mortality'].astype(np.float32)  # Raw mortality (not log1p)
    
    try:
        sev_model = sm.GLM(
            y_sev_pos, 
            X_sev_pos,
            family=sm.families.Tweedie(var_power=1.5, link=sm.families.links.Log())
        ).fit(maxiter=150)
        hb.log(f'    Severity converged: {sev_model.converged}')
        hb.log(f'    Deviance on mortality: {float(sev_model.deviance):.6f}')
    except Exception as e:
        raise RuntimeError(f'ERROR fitting severity (Tweedie GLM) model: {type(e).__name__}: {e}')
    
    sev_summary_path = os.path.join(p.cur_dir, 'severity_model_summary.txt')
    with open(sev_summary_path, 'w') as f:
        f.write('HURDLE MODEL (Two-stage approach for zero-inflated mortality)\n')
        f.write('=' * 80 + '\n\n')
        f.write('Stage 1: Hurdle (Logit) - P(mortality > 0 | landslide=1, X)\n')
        f.write('-' * 80 + '\n')
        f.write(str(hurdle_model.summary()))
        f.write('\n\n')
        f.write('Stage 2: Severity (Tweedie GLM with log link) - mortality > 0, X\n')
        f.write('-' * 80 + '\n')
        f.write(str(sev_model.summary()))
    hb.log(f'  Saved: {sev_summary_path}')
    
    sev_json_path = os.path.join(p.cur_dir, 'severity_model.json')
    sev_output = {
        'model_key': 'hurdle_logit_tweedie',
        'family': 'hurdle_zero_inflated',
        'model_type': 'two_stage_hurdle',
        'fitting_method': 'logit_tweedie_glm',
        'dep_var': 'mortality',
        'dep_var_stage1': 'mortality > 0',
        'dep_var_stage2': 'mortality | mortality > 0',
        'stage1_name': 'hurdle',
        'stage2_name': 'severity',
        'stage1_description': 'Logit for P(mortality > 0)',
        'stage2_description': 'Tweedie GLM (var_power=1.5, link=Log) for E[mortality | >0]',
        'hurdle_converged': bool(hurdle_model.converged),
        'severity_converged': bool(sev_model.converged),
        'n_obs': int(len(df_sev)),
        'n_positive': int(n_sev_pos),
        'hurdle_n_events': int(n_sev_pos),
        'severity_n_obs': int(len(df_sev_pos)),
        'hurdle_dep_mean': float(y_hurdle.mean()),
        'hurdle_llf': float(hurdle_model.llf) if np.isfinite(hurdle_model.llf) else None,
        'hurdle_llnull': float(getattr(hurdle_model, 'llnull', np.nan)) if np.isfinite(getattr(hurdle_model, 'llnull', np.nan)) else None,
        'hurdle_pseudo_r2_mcfadden': (
            float(1.0 - (hurdle_model.llf / hurdle_model.llnull))
            if hasattr(hurdle_model, 'llnull') and np.isfinite(hurdle_model.llf) and np.isfinite(hurdle_model.llnull) and hurdle_model.llnull != 0
            else None
        ),
        'hurdle_aic': float(hurdle_model.aic) if hasattr(hurdle_model, 'aic') else None,
        'severity_n_events': int(n_sev_pos),
        'severity_dep_mean': float(y_sev_pos.mean()) if len(y_sev_pos) else None,
        'severity_llf': float(sev_model.llf) if np.isfinite(sev_model.llf) else None,
        'severity_llnull': float(getattr(sev_model, 'llnull', np.nan)) if np.isfinite(getattr(sev_model, 'llnull', np.nan)) else None,
        'severity_pseudo_r2_mcfadden': (
            float(1.0 - (sev_model.llf / sev_model.llnull))
            if hasattr(sev_model, 'llnull') and np.isfinite(sev_model.llf) and np.isfinite(sev_model.llnull) and sev_model.llnull != 0
            else None
        ),
        'severity_aic': float(sev_model.aic) if hasattr(sev_model, 'aic') else None,
        'n_problematic_gaez_zones_dropped': len(problematic_zones),
        'problematic_gaez_zones_dropped': [z[0] for z in problematic_zones],
        'hurdle_predictors': hurdle_predictors,
        'severity_predictors': sev_predictors,
        'gaez_ref_zone': sev_ref_zone,
        'scaler': sev_scaler_info,
        'hurdle_coefficients': hurdle_model.params.to_dict(),
        'hurdle_std_errors': hurdle_model.bse.to_dict(),
        'hurdle_pvalues': hurdle_model.pvalues.to_dict(),
        'severity_coefficients': sev_model.params.to_dict(),
        'severity_std_errors': sev_model.bse.to_dict(),
        'severity_pvalues': sev_model.pvalues.to_dict(),
        'severity_deviance': float(sev_model.deviance) if hasattr(sev_model, 'deviance') else None,
        'severity_aic': float(sev_model.aic) if hasattr(sev_model, 'aic') else None,
    }
    with open(sev_json_path, 'w') as f:
        json.dump(sev_output, f, indent=2)
    hb.log(f'  Saved: {sev_json_path}')
    
    hb.log('\nHazard and severity model estimation complete.')
    hb.log(f'Outputs written to: {p.cur_dir}')

    return p


def test_hazard_model_specs(p):
    """
    Compare multiple mortality model specifications systematically.
    
    Reuses the data prep, propensity scoring, and IPW computation from estimate_hazard_model.
    For each spec, fits a Tweedie(mortality) model with a different set of mortality predictors.
    
    Specs tested:
    1. 'minimal': population + rain_max_daily only
    2. 'current_coarse': population + rain_max_daily + forest_share + othernat_share (current)
    3. 'expanded_coarse': current + cropland_share + grassland_share (K-1 for coarse LULC)
    4. 'refined_forest_crops': forest_dense_share_refined + forest_open_share_refined + 
                               cropland_rainfed_share_refined + cropland_irrigated_share_refined + grassland_share_refined
    5. 'refined_all_but_sparse': all 8 refined categories minus one baseline (K-1 for refined LULC)
    6. 'current_with_interactions': current + interaction terms (rain × slope, rain × forest)
    
    Output: comparison_table.csv with AIC, pseudo-R², convergence, n_predictors for each spec.
    Individual outputs saved to p.cur_dir/{spec_name}/ subdirectories.
    """
    if not p.run_this:
        return p

def tile_zones(p):
    """
    Generate tile zones for parallel processing of predictions.
    Tiles are defined by offsets and sizes, filtered to include only land-containing tiles.
    Saves tile definitions to blocks_list.csv and sets up iterator replacements.
    """
    if not p.run_this:
        return p

    blocks_list_path = os.path.join(p.cur_dir, 'blocks_list.csv')

    # Check if blocks list already exists
    if os.path.exists(blocks_list_path):
        hb.log('Blocks list already exists, loading from file...')
        blocks_df = pd.read_csv(blocks_list_path, header=None)
        blocks_df.columns = ['col_offset', 'row_offset', 'n_cols', 'n_rows']
        blocks_list = blocks_df.values.tolist()
        hb.log(f'Loaded {len(blocks_list)} tiles from existing blocks_list.csv')
    else:
        hb.log('Creating tile list from input raster...')

        # Get raster info
        # Use a reference raster for tiling (e.g., population raster)
        raster_path = os.path.join(p.base_data_dir, 'worldpop', 'ppp_2000_1km_Aggregated.tif')
        ds = gdal.Open(raster_path)
        n_cols = ds.RasterXSize
        n_rows = ds.RasterYSize
        band = ds.GetRasterBand(1)
        wp_ndv = band.GetNoDataValue()

        p.tile_size = getattr(p, 'processing_resolution', 2000)

        # Generate tile boundaries, filtering out ocean-only tiles
        blocks_list = []
        for row_offset in range(0, n_rows, p.tile_size):
            for col_offset in range(0, n_cols, p.tile_size):
                actual_n_cols = min(p.tile_size, n_cols - col_offset)
                actual_n_rows = min(p.tile_size, n_rows - row_offset)

                # Quick check: does tile have land?
                wp_tile = band.ReadAsArray(col_offset, row_offset, actual_n_cols, actual_n_rows)
                land_mask = np.isfinite(wp_tile)
                if wp_ndv is not None:
                    if abs(wp_ndv) > 1e30:
                        land_mask &= (wp_tile != wp_ndv)
                land_mask &= (wp_tile >= 0)

                if land_mask.sum() > 0:
                    blocks_list.append([col_offset, row_offset, actual_n_cols, actual_n_rows])

        ds = None

        # Save blocks list
        blocks_df = pd.DataFrame(blocks_list, columns=['col_offset', 'row_offset', 'n_cols', 'n_rows'])
        blocks_df.to_csv(blocks_list_path, index=False, header=False)
        hb.log(f'Created {len(blocks_list)} land tiles (filtered ocean tiles)')
        hb.log(f'Blocks list saved to: {blocks_list_path}')

    # Setup iterator for next task
    p.iterator_replacements = {
        'tile_col_offset': [block[0] for block in blocks_list],
        'tile_row_offset': [block[1] for block in blocks_list],
        'tile_n_cols': [block[2] for block in blocks_list],
        'tile_n_rows': [block[3] for block in blocks_list],
        'cur_dir_parent_dir': [
            os.path.join(p.cur_dir, f'{block[1]}_{block[0]}')
            for block in blocks_list
        ]
    }
    hb.log(f'Set up iterator replacements for {len(blocks_list)} tiles.')
    return p


def _load_hazard_coefs(p):
    """Load hazard model coefficients JSON and return (coefs, scaler, ref_zone)."""
    coef_path = os.path.join(p.estimate_hazard_model_dir,
                             'hazard_model.json')
    with open(coef_path) as f:
        coef_data = json.load(f)
    coefs    = coef_data['coefficients']
    scaler   = coef_data['scaler']
    ref_zone = int(coef_data['gaez_ref_zone'])
    intercept_correction = float(coef_data.get('intercept_correction', 0.0) or 0.0)
    if np.isfinite(intercept_correction):
        coefs['const'] = float(coefs.get('const', 0.0) + intercept_correction)
    return coefs, scaler, ref_zone


def _load_mortality_coefs(p):
    """Load mortality model (Stage 2) coefficients JSON and return (coefs, scaler, ref_zone)."""
    coef_path = os.path.join(p.estimate_hazard_model_dir,
                             'mortality_model.json')
    with open(coef_path) as f:
        coef_data = json.load(f)
    return (coef_data['coefficients'], coef_data['scaler'], 
            int(coef_data['gaez_ref_zone']))


def _load_severity_coefs(p):
    """Load severity model (conditional mortality E[Y|L=1]) JSON and return (coefs, scaler, ref_zone).
    
    For Hurdle-Tweedie: coefs contains both 'hurdle_coefficients' and 'severity_coefficients'
    For OLS: coefs contains 'severity_coefficients' only
    """
    coef_path = os.path.join(p.estimate_hazard_model_dir,
                             'severity_model.json')
    with open(coef_path) as f:
        coef_data = json.load(f)
    # Return entire coef_data so _predict_severity_tile can access all coefficient types
    return (coef_data, coef_data.get('scaler', {}), 
            int(coef_data.get('gaez_ref_zone', 10)))


def _predict_hazard_tile(tile_data, coefs, scaler, ref_zone):
    """
    Apply logit hazard model to a dict of tile arrays.

    Parameters
    ----------
    tile_data : dict with keys matching predictor names, values are 2-D arrays
    coefs     : dict of corrected coefficients from JSON
    scaler    : dict of {col: {mean, std}} for continuous vars and LULC centering
    ref_zone  : int, the GAEZ zone used as reference (its dummy is omitted)

    Returns
    -------
    prob : float32 array, shape = tile shape, Pr(landslide)
    """
    shape = next(iter(tile_data.values())).shape
    eta   = np.full(shape, coefs.get('const', 0.0), dtype=np.float64)

    # Continuous predictors — standardize using training stats.
    # Apply training winsor caps first (if present), then z-score guardrails.
    Z_CAP = 8.0
    for col in [
        'slope_degree', 'roughness', 'tpi', 'elev_stdev', 'road_density', 'dist_to_fault_km',
        'deforestation_forest_share_1yr', 'deforestation_forest_share_3yr',
    ]:
        arr = tile_data[col].astype(np.float64)

        # Training-consistent winsor cap (present in new model JSONs).
        cap = scaler.get(col, {}).get('cap', None)
        if cap is not None and np.isfinite(cap):
            if col in {'slope_degree', 'roughness', 'elev_stdev', 'road_density', 'dist_to_fault_km'}:
                arr = np.clip(arr, 0.0, float(cap))
            else:
                arr = np.minimum(arr, float(cap))
        else:
            # Legacy fallback for older model JSONs without cap metadata.
            if col == 'road_density':
                arr = np.clip(arr, 0.0, 10.0)
            else:
                arr = np.clip(arr, -1e6, 1e6)
        
        mu    = scaler[col]['mean']
        sigma = scaler[col]['std']
        if sigma > 0:
            arr = (arr - mu) / sigma
            arr = np.clip(arr, -Z_CAP, Z_CAP)
        eta += coefs.get(col, 0.0) * arr

    # GAEZ dummies — add coefficient for each zone != ref_zone
    gaez_arr = tile_data['gaez_zone'].astype(np.int16)
    for col, beta in coefs.items():
        if not col.startswith('gaez_'):
            continue
        zone = int(col.split('_')[1])
        if zone == ref_zone:
            continue
        eta += beta * (gaez_arr == zone).astype(np.float64)

    # Logit inverse link: p = 1 / (1 + exp(-eta))
    eta  = np.clip(eta, -500, 500)
    prob = (1.0 / (1.0 + np.exp(-eta))).astype(np.float32)
    return prob


def _predict_mortality_tile(tile_data, mort_coefs, mort_scaler, ref_zone):
    """
    Compute IPW-adjusted expected mortality (marginal over hazard probability).

    Mortality model (Stage 2, IPW-weighted):
        E[deaths | X] = exp(γ0 + γ1·log1p(pop)_std + γ2·rain_std 
                              + γ3·forest_centered + γ4·othernat_centered
                              + Σ γ_z·(gaez_zone == z))
    where standardization/centering uses training stats.
    
    IPW weighting corrects for selection bias (deaths observed only where hazards occur),
    allowing prediction of marginal mortality for the full population.
    
    NOTE: Intercept correction from sampling is NOT applied here. Pixel-level predictions
    are unconditional and population-representative. Intercept correction should only be
    applied at zonal aggregation time if needed for summary statistics.
    """
    shape = next(iter(tile_data.values())).shape
    eta_m = np.full(shape, mort_coefs.get('const', 0.0), dtype=np.float64)
    Z_CAP = 8.0

    # Standardized continuous predictors in the refined Stage 2 model
    feature_specs = [
        ('population_log1p', 'population', np.log1p),
        ('rain_max_daily', 'rain_max_daily', None),
        ('slope_degree', 'slope_degree', None),
        ('road_density', 'road_density', None),
    ]
    for feature_key, raster_key, transform_fn in feature_specs:
        if feature_key == 'population_log1p' and 'population_log1p' not in mort_coefs:
            feature_key = 'population'
            transform_fn = None
        if raster_key not in tile_data:
            continue
        if feature_key not in mort_scaler:
            if feature_key != 'population_log1p' or 'population' not in mort_scaler:
                continue

        arr = tile_data[raster_key].astype(np.float64)
        if transform_fn is not None:
            arr = transform_fn(np.clip(arr, 0.0, None))

        # Training-consistent winsor cap (present in new model JSONs).
        cap = mort_scaler.get(feature_key, {}).get('cap', None)
        if cap is not None and np.isfinite(cap):
            arr = np.clip(arr, 0.0, float(cap))
        elif feature_key == 'road_density':
            # Legacy fallback for older model JSONs without cap metadata.
            arr = np.clip(arr, 0.0, 10.0)
        
        scaler_key = feature_key if feature_key in mort_scaler else 'population'
        mu    = mort_scaler[scaler_key]['mean']
        sigma = mort_scaler[scaler_key]['std']
        if sigma > 0:
            arr = (arr - mu) / sigma
            arr = np.clip(arr, -Z_CAP, Z_CAP)
        else:
            arr = arr - mu
        eta_m += mort_coefs.get(feature_key, mort_coefs.get('population', 0.0) if feature_key == 'population_log1p' else 0.0) * arr

    # Centered vegetation shares in the refined Stage 2 model.
    # Use training scaler means when available; fall back to coarse scaler
    # entries or zero if missing so counterfactual transformations have effect.
    for col in ['forest_open_share_refined', 'forest_dense_share_refined', 'grassland_share_refined']:
        if col not in tile_data:
            continue
        arr = np.clip(tile_data[col].astype(np.float64), 0.0, 1.0)
        # Prefer exact scaler mean; try sensible coarse fallbacks; default to 0.0
        if col in mort_scaler:
            mu = mort_scaler[col].get('mean', 0.0)
        else:
            if col.startswith('forest') and 'forest_share' in mort_scaler:
                mu = mort_scaler['forest_share'].get('mean', 0.0)
            elif col == 'grassland_share_refined' and 'grassland_share' in mort_scaler:
                mu = mort_scaler['grassland_share'].get('mean', 0.0)
            else:
                mu = 0.0
        arr = arr - mu  # Centering only, no std division
        eta_m += mort_coefs.get(col, 0.0) * arr

    # Backward-compatible fallback if a legacy coarse model is loaded.
    for col in ['forest_share', 'othernat_share']:
        if col not in tile_data or col not in mort_scaler:
            continue
        arr = tile_data[col].astype(np.float64)
        mu  = mort_scaler[col]['mean']
        eta_m += mort_coefs.get(col, 0.0) * (arr - mu)

    # Log link inverse: E[Y] = exp(eta)
    eta_m         = np.clip(eta_m, -500, 500)
    exp_mortality = np.exp(eta_m).astype(np.float32)
    return exp_mortality


def _load_tile_data(p, year, col_offset, row_offset, n_cols, n_rows, ref_shape,
                    include_hazard_covariates=False):
    """
    Load all raster inputs for one tile-year. Returns dict of float32 arrays
    or None if any required file is missing or fails to load.
    
    All keys MUST be present and valid for function to succeed.
    """
    cache_root = os.path.join(
        getattr(p, 'project_dir', p.cur_dir),
        'intermediate', 'build_estimation_table', 'static_raster_cache'
    )

    paths = {
        'slope_degree': os.path.join(*p.base_data, 'preprocess_data', 'geomorpho90m',
                                     'slope_geomorpho.tif'),
        'roughness':    os.path.join(*p.base_data, 'preprocess_data', 'geomorpho90m',
                                     'roughness_geomorpho.tif'),
        'tpi':          os.path.join(*p.base_data, 'preprocess_data', 'geomorpho90m',
                                     'tpi_geomorpho.tif'),
        'elev_stdev':   os.path.join(*p.base_data, 'preprocess_data', 'geomorpho90m',
                                     'elev_stdev_geomorpho.tif'),
        'gaez_zone':    os.path.join(*p.base_data, 'preprocess_data', 'fao_gaez',
                                     'fao_gaez.tif'),
        'rain_max_daily': os.path.join(*p.base_data, 'preprocess_data', 'era5_land',
                                       f'era5_max_daily_mm_{year}.tif'),
        'population':   os.path.join(*p.base_data, 'preprocess_data', 'landscan',
                         f'landscan_{year}_1km.tif'),
        'road_density': os.path.join(*p.base_data, 'preprocess_data', 'grip_roads',
                                     'road_density_km_per_km2.tif'),
    }
    if include_hazard_covariates:
        paths['dist_to_fault_km'] = os.path.join(
            *p.base_data, 'preprocess_data', 'gem_faults',
            'distance_to_fault_km.tif'
        )
        paths['deforestation_forest_share_1yr'] = os.path.join(
            *p.base_data, 'preprocess_data', 'deforestation', 'coarse',
            f'deforestation_forest_share_1yr_{year}.tif'
        )
        paths['deforestation_forest_share_3yr'] = os.path.join(
            *p.base_data, 'preprocess_data', 'deforestation', 'coarse',
            f'deforestation_forest_share_3yr_{year}.tif'
        )
        # Also load forest_share for scenario counterfactuals
        paths['forest_share'] = os.path.join(
            *p.base_data, 'preprocess_data', 'esa',
            f'esacci_share_forest_{year}.tif'
        )
    lulc_vars = ['forest_dense', 'forest_open', 'grassland'] # 'bare_areas', 'shrubland', 'sparse' not in mortality model
    for var in lulc_vars:
        paths[f'{var}_share_refined'] = os.path.join(*p.base_data, 'preprocess_data', 'esa_refined',
                                                      f'esacci_share_refined_{var}_{year}.tif')

    required_keys = set(paths.keys())
    tile_data = {}
    failed_keys = []
    
    static_keys = {
        'slope_degree', 'roughness', 'tpi', 'elev_stdev', 'gaez_zone',
        'road_density', 'dist_to_fault_km',
    }
    annual_keys = {
        'rain_max_daily', 'population',
        'deforestation_forest_share_1yr', 'deforestation_forest_share_3yr',
        'forest_share',
        'forest_dense_share_refined', 'forest_open_share_refined', 'grassland_share_refined',
    }

    for key, path in paths.items():
        try:
            if key in static_keys:
                local_path = _get_cached_estimation_table_path(path, cache_root, prefer_year=False)
            elif key in annual_keys:
                local_path = _get_cached_estimation_table_path(path, cache_root, year=year, prefer_year=True)
            else:
                local_path = _get_cached_estimation_table_path(path, cache_root, year=year)

            if local_path is None:
                local_path = _get_or_cache_local_path(path, cache_root)
            ds   = gdal.Open(local_path)
            if ds is None:
                hb.log(f'    {key}: GDAL could not open at {local_path}')
                failed_keys.append(key)
                continue
            
            band = ds.GetRasterBand(1)
            if band is None:
                hb.log(f'    {key}: could not get band 1')
                ds = None
                failed_keys.append(key)
                continue
            
            arr = band.ReadAsArray(col_offset, row_offset, n_cols, n_rows)
            ds = None
            
            # Validate array
            if arr is None:
                hb.log(f'    {key}: ReadAsArray returned None')
                failed_keys.append(key)
                continue
            
            if arr.size == 0:
                hb.log(f'    {key}: empty array (size=0)')
                failed_keys.append(key)
                continue
            
            # Check shape - handle edge tiles
            if arr.ndim != 2:
                hb.log(f'    {key}: unexpected ndim {arr.ndim} (expected 2)')
                failed_keys.append(key)
                continue
            
            if arr.shape != (n_rows, n_cols):
                hb.log(f'    {key}: shape {arr.shape} vs expected ({n_rows},{n_cols})')
                # Pad edge tiles with NaN
                if arr.shape[0] <= n_rows and arr.shape[1] <= n_cols:
                    padded = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
                    padded[:arr.shape[0], :arr.shape[1]] = arr
                    arr = padded
                    hb.log(f'    {key}: padded with NaN')
                else:
                    hb.log(f'    {key}: shape too large, cannot pad')
                    failed_keys.append(key)
                    continue
            
            # If this is the GAEZ zone raster, map original zones into AEZ groups
            if key == 'gaez_zone':
                try:
                    arr_int = arr.astype(np.int16)
                    mapped = np.zeros_like(arr_int, dtype=np.int16)
                    valid_mask = (arr_int >= 0) & (arr_int < len(_GAEZ_TO_GROUP))
                    if valid_mask.any():
                        mapped[valid_mask] = _GAEZ_TO_GROUP[arr_int[valid_mask]]
                    arr = mapped
                except Exception:
                    arr = arr.astype(np.int16)

            tile_data[key] = arr.astype(np.float32)
            
        except Exception as e:
            hb.log(f'    {key}: {type(e).__name__}: {str(e)[:100]}')
            failed_keys.append(key)
            continue

    # Strict: if ANY key failed, reject entire tile
    if failed_keys:
        hb.log(f'  Tile {row_offset},{col_offset} year {year}: '
               f'{len(failed_keys)}/{len(required_keys)} failed ({failed_keys}) — skipping year.')
        return None
    
    # Final validation: all keys must be present
    missing = required_keys - set(tile_data.keys())
    if missing:
        hb.log(f'  Tile {row_offset},{col_offset} year {year}: '
               f'internal error: missing keys {missing} — skipping year.')
        return None

    return tile_data


def _write_tile_raster(arr, out_path, ref_ds, col_offset, row_offset):
    """Write a float32 tile array to a GeoTIFF at the correct spatial offset."""
    gt      = list(ref_ds.GetGeoTransform())
    gt[0]  += col_offset * gt[1]   # shift x origin
    gt[3]  += row_offset * gt[5]   # shift y origin

    driver  = gdal.GetDriverByName('GTiff')
    n_rows, n_cols = arr.shape
    ds_out  = driver.Create(out_path, n_cols, n_rows, 1, gdal.GDT_Float32,
                            options=['COMPRESS=LZW', 'TILED=YES'])
    ds_out.SetGeoTransform(gt)
    ds_out.SetProjection(ref_ds.GetProjection())
    band = ds_out.GetRasterBand(1)
    band.SetNoDataValue(-9999.0)  # Use numeric sentinel, not float('nan')
    # Replace NaN with NoDataValue
    arr_clean = np.where(np.isnan(arr), -9999.0, arr)
    band.WriteArray(arr_clean)
    band.FlushCache()
    ds_out = None


def _build_tile_land_mask(tile_data):
    """Build a conservative land mask for tile outputs using GAEZ zone IDs."""
    if 'gaez_zone' not in tile_data:
        shape = next(iter(tile_data.values())).shape
        return np.ones(shape, dtype=bool)

    gaez_arr = tile_data['gaez_zone']
    # Keep only valid GAEZ classes used in training (1..57); everything else is ocean/NoData.
    land_mask = np.isfinite(gaez_arr) & (gaez_arr >= 1) & (gaez_arr <= 57)

    # Also require basic terrain validity when available.
    if 'slope_degree' in tile_data:
        land_mask &= np.isfinite(tile_data['slope_degree'])
    if 'roughness' in tile_data:
        land_mask &= np.isfinite(tile_data['roughness'])

    return land_mask


def predict_landslides_observed(p):
    """
    Predict landslide hazard probability with observed deforestation.
    
    Outputs per year: observed_landslide_prob_{year}.tif
    """
    if not p.run_this:
        return p

    col_offset = int(p.tile_col_offset)
    row_offset = int(p.tile_row_offset)
    n_cols     = int(p.tile_n_cols)
    n_rows     = int(p.tile_n_rows)

    coefs, scaler, ref_zone = _load_hazard_coefs(p)

    raster_size = p.reference_raster_info['raster_size']
    ref_shape   = (raster_size[1], raster_size[0])

    ref_path = os.path.join(*p.base_data, 'preprocess_data', 'geomorpho90m',
                            'slope_geomorpho.tif')
    vsis3_ref_path = s3_handler.get_vsis3_path(ref_path)
    ref_ds   = gdal.Open(vsis3_ref_path)

    required_keys = {
        'slope_degree', 'roughness', 'tpi', 'elev_stdev', 'gaez_zone',
        'road_density', 'dist_to_fault_km',
        'deforestation_forest_share_1yr', 'deforestation_forest_share_3yr',
    }

    for year in p.prediction_years:
        out_path = os.path.join(p.cur_dir, f'observed_landslide_prob_{year}.tif')
        if os.path.exists(out_path) and not getattr(p, 'force_run', False):
            continue
        if os.path.exists(out_path) and getattr(p, 'force_run', False):
            os.remove(out_path)

        tile_data = _load_tile_data(
            p, year, col_offset, row_offset, n_cols, n_rows, ref_shape,
            include_hazard_covariates=True
        )
        if tile_data is None:
            continue

        missing_keys = required_keys - set(tile_data.keys())
        if missing_keys:
            hb.log(f'  Tile {row_offset},{col_offset} year {year}: '
                   f'missing keys {missing_keys} — skipping year.')
            continue

        try:
            prob = _predict_hazard_tile(tile_data, coefs, scaler, ref_zone)
            land_mask = _build_tile_land_mask(tile_data)
            prob = np.where(land_mask, prob, np.nan).astype(np.float32)

            valid_mask = ~np.isnan(prob) & ~np.isinf(prob)
            n_valid = valid_mask.sum()
            if n_valid > 0:
                hb.log(f'    Observed: {n_valid:,} valid pixels | '
                       f'mean={np.nanmean(prob[valid_mask]):.6e}')
            else:
                hb.log(f'    WARNING: All predictions are NaN/inf!')

            _write_tile_raster(prob, out_path, ref_ds, col_offset, row_offset)
        except Exception as e:
            hb.log(f'  Tile {row_offset},{col_offset} year {year}: '
                   f'exception: {type(e).__name__}: {e} — skipping.')
            continue

    ref_ds = None
    return p


def predict_landslides_no_deforestation(p):
    """
    Predict landslide hazard probability with zero deforestation everywhere.
    
    Outputs per year: no_deforestation_landslide_prob_{year}.tif
    """
    if not p.run_this:
        return p

    col_offset = int(p.tile_col_offset)
    row_offset = int(p.tile_row_offset)
    n_cols     = int(p.tile_n_cols)
    n_rows     = int(p.tile_n_rows)

    coefs, scaler, ref_zone = _load_hazard_coefs(p)

    raster_size = p.reference_raster_info['raster_size']
    ref_shape   = (raster_size[1], raster_size[0])

    ref_path = os.path.join(*p.base_data, 'preprocess_data', 'geomorpho90m',
                            'slope_geomorpho.tif')
    vsis3_ref_path = s3_handler.get_vsis3_path(ref_path)
    ref_ds   = gdal.Open(vsis3_ref_path)

    required_keys = {
        'slope_degree', 'roughness', 'tpi', 'elev_stdev', 'gaez_zone',
        'road_density', 'dist_to_fault_km',
        'deforestation_forest_share_1yr', 'deforestation_forest_share_3yr',
    }

    for year in p.prediction_years:
        out_path = os.path.join(p.cur_dir, f'no_deforestation_landslide_prob_{year}.tif')
        if os.path.exists(out_path) and not getattr(p, 'force_run', False):
            continue
        if os.path.exists(out_path) and getattr(p, 'force_run', False):
            os.remove(out_path)

        tile_data = _load_tile_data(
            p, year, col_offset, row_offset, n_cols, n_rows, ref_shape,
            include_hazard_covariates=True
        )
        if tile_data is None:
            continue

        missing_keys = required_keys - set(tile_data.keys())
        if missing_keys:
            hb.log(f'  Tile {row_offset},{col_offset} year {year}: '
                   f'missing keys {missing_keys} — skipping year.')
            continue

        try:
            # Set deforestation to zero
            tile_no_defor = {k: v.copy() for k, v in tile_data.items()}
            tile_no_defor['deforestation_forest_share_1yr'] = np.zeros_like(tile_data['deforestation_forest_share_1yr'])
            tile_no_defor['deforestation_forest_share_3yr'] = np.zeros_like(tile_data['deforestation_forest_share_3yr'])
            
            prob = _predict_hazard_tile(tile_no_defor, coefs, scaler, ref_zone)
            land_mask = _build_tile_land_mask(tile_data)
            prob = np.where(land_mask, prob, np.nan).astype(np.float32)

            valid_mask = ~np.isnan(prob) & ~np.isinf(prob)
            n_valid = valid_mask.sum()
            if n_valid > 0:
                hb.log(f'    No-deforestation: {n_valid:,} valid pixels | '
                       f'mean={np.nanmean(prob[valid_mask]):.6e}')
            else:
                hb.log(f'    WARNING: All predictions are NaN/inf!')

            _write_tile_raster(prob, out_path, ref_ds, col_offset, row_offset)
        except Exception as e:
            hb.log(f'  Tile {row_offset},{col_offset} year {year}: '
                   f'exception: {type(e).__name__}: {e} — skipping.')
            continue

    ref_ds = None
    return p


def predict_landslides_scenarios(p):
    """
    Predict landslide hazard probability under scenario deforestation rates.
    
    Outputs per scenario and year: scenario_{name}_landslide_prob_{year}.tif
    """
    if not p.run_this:
        return p

    col_offset = int(p.tile_col_offset)
    row_offset = int(p.tile_row_offset)
    n_cols     = int(p.tile_n_cols)
    n_rows     = int(p.tile_n_rows)

    coefs, scaler, ref_zone = _load_hazard_coefs(p)

    raster_size = p.reference_raster_info['raster_size']
    ref_shape   = (raster_size[1], raster_size[0])

    ref_path = os.path.join(*p.base_data, 'preprocess_data', 'geomorpho90m',
                            'slope_geomorpho.tif')
    vsis3_ref_path = s3_handler.get_vsis3_path(ref_path)
    ref_ds   = gdal.Open(vsis3_ref_path)

    required_keys = {
        'slope_degree', 'roughness', 'tpi', 'elev_stdev', 'gaez_zone',
        'road_density', 'dist_to_fault_km',
        'deforestation_forest_share_1yr', 'deforestation_forest_share_3yr',
        'forest_share',
    }

    scenarios = getattr(p, 'forest_value_scenarios', [
        ('5pct_annual', 0.05, 0.15),
        ('10pct_annual', 0.10, 0.30),
    ])

    for year in p.prediction_years:
        for scenario_name, rate_1yr, rate_3yr in scenarios:
            out_path = os.path.join(p.cur_dir, f'scenario_{scenario_name}_landslide_prob_{year}.tif')
            if os.path.exists(out_path) and not getattr(p, 'force_run', False):
                continue
            if os.path.exists(out_path) and getattr(p, 'force_run', False):
                os.remove(out_path)

            tile_data = _load_tile_data(
                p, year, col_offset, row_offset, n_cols, n_rows, ref_shape,
                include_hazard_covariates=True
            )
            if tile_data is None:
                continue

            missing_keys = required_keys - set(tile_data.keys())
            if missing_keys:
                hb.log(f'  Tile {row_offset},{col_offset} year {year} scenario {scenario_name}: '
                       f'missing keys {missing_keys} — skipping.')
                continue

            try:
                # Build counterfactual deforestation
                tile_scenario = {k: v.copy() for k, v in tile_data.items()}
                tile_scenario['deforestation_forest_share_1yr'] = (
                    tile_data['forest_share'].astype(np.float32) * rate_1yr
                ).clip(0, 1)
                tile_scenario['deforestation_forest_share_3yr'] = (
                    tile_data['forest_share'].astype(np.float32) * rate_3yr
                ).clip(0, 1)
                
                prob = _predict_hazard_tile(tile_scenario, coefs, scaler, ref_zone)
                land_mask = _build_tile_land_mask(tile_data)
                prob = np.where(land_mask, prob, np.nan).astype(np.float32)

                valid_mask = ~np.isnan(prob) & ~np.isinf(prob)
                n_valid = valid_mask.sum()
                if n_valid > 0:
                    hb.log(f'    Scenario {scenario_name}: {n_valid:,} valid pixels | '
                           f'mean={np.nanmean(prob[valid_mask]):.6e}')
                else:
                    hb.log(f'    WARNING: All predictions are NaN/inf!')

                _write_tile_raster(prob, out_path, ref_ds, col_offset, row_offset)
            except Exception as e:
                hb.log(f'  Tile {row_offset},{col_offset} year {year} scenario {scenario_name}: '
                       f'exception: {type(e).__name__}: {e} — skipping.')
                continue

    ref_ds = None
    return p


def predict_mortality(p):
    """
    Predict expected mortality using severity model given observed covariates.
    
    Outputs per year: expected_mortality_{year}.tif
    Uses: E[mortality | landslide=1, observed covariates] from severity model
    """
    if not p.run_this:
        return p

    col_offset = int(p.tile_col_offset)
    row_offset = int(p.tile_row_offset)
    n_cols     = int(p.tile_n_cols)
    n_rows     = int(p.tile_n_rows)

    sev_coefs, sev_scaler, sev_ref_zone = _load_severity_coefs(p)

    raster_size = p.reference_raster_info['raster_size']
    ref_shape   = (raster_size[1], raster_size[0])

    ref_path = os.path.join(*p.base_data, 'preprocess_data', 'geomorpho90m',
                            'slope_geomorpho.tif')
    vsis3_ref_path = s3_handler.get_vsis3_path(ref_path)
    ref_ds   = gdal.Open(vsis3_ref_path)

    required_keys = {
        'gaez_zone', 'population', 'rain_max_daily', 'slope_degree', 'road_density',
    }

    for year in p.prediction_years:
        out_path = os.path.join(p.cur_dir, f'expected_mortality_{year}.tif')
        if os.path.exists(out_path) and not getattr(p, 'force_run', False):
            continue
        if os.path.exists(out_path) and getattr(p, 'force_run', False):
            os.remove(out_path)

        tile_data = _load_tile_data(
            p, year, col_offset, row_offset, n_cols, n_rows, ref_shape,
            include_hazard_covariates=False
        )
        if tile_data is None:
            continue

        missing_keys = required_keys - set(tile_data.keys())
        if missing_keys:
            hb.log(f'  Tile {row_offset},{col_offset} year {year}: '
                   f'missing keys {missing_keys} — skipping year.')
            continue

        try:
            severity = _predict_severity_tile(tile_data, sev_coefs, sev_scaler, sev_ref_zone)
            land_mask = _build_tile_land_mask(tile_data)
            pop_mask = (tile_data['population'] > 0)
            final_mask = land_mask & pop_mask
            severity = np.where(final_mask, severity, np.nan).astype(np.float32)

            valid_mask = ~np.isnan(severity) & ~np.isinf(severity)
            n_valid = valid_mask.sum()
            if n_valid > 0:
                hb.log(f'    Mortality: {n_valid:,} valid pixels | '
                       f'mean={np.nanmean(severity[valid_mask]):.6e}')
            else:
                hb.log(f'    WARNING: All predictions are NaN/inf!')

            _write_tile_raster(severity, out_path, ref_ds, col_offset, row_offset)
        except Exception as e:
            hb.log(f'  Tile {row_offset},{col_offset} year {year}: '
                   f'exception: {type(e).__name__}: {e} — skipping.')
            continue

    ref_ds = None
    return p


def _predict_severity_tile(tile_data, sev_coefs, sev_scaler, sev_ref_zone):
    """
    Compute conditional severity (expected mortality | landslide=1) for a tile.
    Uses hurdle model: E[mortality] = P(mortality > 0) × E[exp(log1p(mortality)) | >0]
    
    Two-stage prediction:
    1. Hurdle: logit for P(mortality > 0)
    2. Severity: OLS on log1p(mortality) for positive-only rows
    
    sev_coefs should contain both 'hurdle_coefficients' and 'severity_coefficients'
    """
    shape = next(iter(tile_data.values())).shape
    
    # Standardized continuous predictors (same for both stages)
    feature_specs = [
        ('population_log1p', 'population', np.log1p),
        ('rain_max_daily', 'rain_max_daily', None),
        ('slope_degree', 'slope_degree', None),
        ('road_density', 'road_density', None),
    ]
    
    # Build predictor arrays
    X_features = {}
    for feature_key, raster_key, transform_fn in feature_specs:
        if raster_key not in tile_data:
            continue
        if feature_key not in sev_scaler:
            continue
        
        arr = tile_data[raster_key].astype(np.float64)
        if transform_fn is not None:
            arr = transform_fn(np.clip(arr, 0.0, None))
        
        mu = sev_scaler[feature_key]['mean']
        sigma = sev_scaler[feature_key]['std']
        if sigma > 0:
            arr = (arr - mu) / sigma
        else:
            arr = arr - mu
        
        X_features[feature_key] = arr
    
    # Stage 1: Hurdle (Logit) - P(mortality > 0)
    eta_hurdle = np.full(shape, sev_coefs.get('hurdle_coefficients', {}).get('const', 0.0), dtype=np.float64)
    
    for feature_key in feature_specs:
        fname = feature_key[0]
        if fname in X_features:
            coef_val = sev_coefs.get('hurdle_coefficients', {}).get(f'std_{fname}', 0.0)
            eta_hurdle += coef_val * X_features[fname]
    
    # Add GAEZ effects for hurdle
    if 'gaez_zone' in tile_data:
        gaez_arr = tile_data['gaez_zone'].astype(np.int16)
        for z in set(np.unique(gaez_arr)):
            if z > 0 and z != sev_ref_zone:
                col = f'gaez_{int(z)}'
                if col in sev_coefs.get('hurdle_coefficients', {}):
                    eta_hurdle += sev_coefs['hurdle_coefficients'][col] * (gaez_arr == z).astype(np.float64)
    
    # Logit inverse: p = 1 / (1 + exp(-eta))
    p_positive = 1.0 / (1.0 + np.exp(-np.clip(eta_hurdle, -500, 500)))
    
    # Stage 2: Severity (OLS on log1p(mortality))
    eta_severity = np.full(shape, sev_coefs.get('severity_coefficients', {}).get('const', 0.0), dtype=np.float64)
    
    for feature_key in feature_specs:
        fname = feature_key[0]
        if fname in X_features:
            coef_val = sev_coefs.get('severity_coefficients', {}).get(f'std_{fname}', 0.0)
            eta_severity += coef_val * X_features[fname]
    
    # Add GAEZ effects for severity
    if 'gaez_zone' in tile_data:
        gaez_arr = tile_data['gaez_zone'].astype(np.int16)
        for z in set(np.unique(gaez_arr)):
            if z > 0 and z != sev_ref_zone:
                col = f'gaez_{int(z)}'
                if col in sev_coefs.get('severity_coefficients', {}):
                    eta_severity += sev_coefs['severity_coefficients'][col] * (gaez_arr == z).astype(np.float64)
    
    # Tweedie prediction with Log link: predict = exp(eta)
    severity = np.exp(np.clip(eta_severity, -500, 100))  # Clip to prevent overflow
    severity = np.maximum(severity, 0.0)  # Ensure non-negative
    
    # Combine: mortality = P(>0) × E[severity | >0]
    mortality = p_positive * severity
    
    return mortality.astype(np.float32)


def stitch_tiles(p):
    """
    Stitch tile-level predictions into global annual rasters.
    
    Outputs for each year:
    - observed_landslide_prob_{year}.tif       (hazard with observed deforestation)
    - no_deforestation_landslide_prob_{year}.tif (hazard with zero deforestation)
    - scenario_{name}_landslide_prob_{year}.tif  (hazard under each scenario)
    - expected_mortality_{year}.tif            (severity-based mortality expectation)
    
    Forest value can then be computed as post-processing:
    forest_value = (observed_hazard - scenario_hazard) × mortality
    """
    if not p.run_this:
        return p

    # Read tile list
    blocks_list_path = os.path.join(p.tile_zones_dir, 'blocks_list.csv')
    blocks_df        = pd.read_csv(blocks_list_path, header=None)
    blocks_df.columns = ['col_offset', 'row_offset', 'n_cols', 'n_rows']
    blocks_list      = blocks_df.values.tolist()

    raster_size    = p.reference_raster_info['raster_size']
    n_cols_full    = raster_size[0]
    n_rows_full    = raster_size[1]

    # Open reference for projection/geotransform
    ref_path = p.reference_raster_path
    ref_ds   = gdal.Open(ref_path)
    gt       = ref_ds.GetGeoTransform()
    proj     = ref_ds.GetProjection()
    ref_ds   = None

    driver = gdal.GetDriverByName('GTiff')
    NODATAVALUE = -9999.0

    # Define outputs to stitch
    scenario_specs = [
        {
            'name': 'observed_hazard',
            'tile_subdir': 'predict_landslides_observed',
            'tile_filename': 'observed_landslide_prob_{year}.tif',
            'global_filename': 'observed_landslide_prob_{year}.tif',
        },
        {
            'name': 'no_deforestation_hazard',
            'tile_subdir': 'predict_landslides_no_deforestation',
            'tile_filename': 'no_deforestation_landslide_prob_{year}.tif',
            'global_filename': 'no_deforestation_landslide_prob_{year}.tif',
        },
        {
            'name': 'mortality',
            'tile_subdir': 'predict_mortality',
            'tile_filename': 'expected_mortality_{year}.tif',
            'global_filename': 'expected_mortality_{year}.tif',
        },
    ]

    # Add scenario hazard outputs dynamically
    scenarios = getattr(p, 'forest_value_scenarios', [
        ('5pct_annual', 0.05, 0.15),
        ('10pct_annual', 0.10, 0.30),
    ])
    for scenario_name, _, _ in scenarios:
        scenario_specs.append({
            'name': f'scenario_{scenario_name}_hazard',
            'tile_subdir': 'predict_landslides_scenarios',
            'tile_filename': f'scenario_{scenario_name}_landslide_prob_{{year}}.tif',
            'global_filename': f'scenario_{scenario_name}_landslide_prob_{{year}}.tif',
        })

    for year in p.prediction_years:
        for spec in scenario_specs:
            out_path = os.path.join(p.cur_dir,
                                    spec['global_filename'].format(year=year))
            if os.path.exists(out_path) and not getattr(p, 'force_run', False):
                continue
            if os.path.exists(out_path) and getattr(p, 'force_run', False):
                os.remove(out_path)

            hb.log(f'Stitching {spec["name"]} {year}...')
            ds_out = driver.Create(
                out_path, n_cols_full, n_rows_full, 1, gdal.GDT_Float32,
                options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'],
            )
            ds_out.SetGeoTransform(gt)
            ds_out.SetProjection(proj)
            band_out = ds_out.GetRasterBand(1)
            band_out.SetNoDataValue(NODATAVALUE)
            band_out.Fill(NODATAVALUE)
            
            tiles_written = 0
            missing_tiles = []

            for block in blocks_list:
                col_off, row_off, n_c, n_r = [int(x) for x in block]
                tile_dir  = os.path.join(p.tile_zones_dir,
                                         f'{row_off}_{col_off}')
                tile_path = os.path.join(
                    tile_dir,
                    spec['tile_subdir'],
                    spec['tile_filename'].format(year=year)
                )
                if not os.path.exists(tile_path):
                    missing_tiles.append(tile_path)
                    continue

                ds_tile  = gdal.Open(tile_path)
                tile_arr = ds_tile.GetRasterBand(1).ReadAsArray().astype(np.float32)
                ds_tile  = None
                
                # Replace NaN with output NODATAVALUE
                tile_arr = np.where(np.isnan(tile_arr), NODATAVALUE, tile_arr)
                
                # Write tile
                band_out.WriteArray(tile_arr, col_off, row_off)
                tiles_written += 1

            band_out.FlushCache()
            ds_out = None
            if missing_tiles:
                hb.log(
                    f'  Missing {len(missing_tiles)} tile files for {spec["name"]} {year}'
                )
            hb.log(f'  Wrote {tiles_written} tiles to {os.path.basename(out_path)}')

    hb.log('Stitching complete.')
    return p

