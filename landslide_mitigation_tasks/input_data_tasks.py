import os
import sys
import zipfile

import hazelbean as hb
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from osgeo import gdal, ogr, osr
import pygeoprocessing as pygeo
from pyproj import Transformer
from shapely.geometry import Point
import subprocess
import shutil

from .utils import s3_handler


def preprocess_data(p):
    """
    Creates unified dir for preprocessed data
    """
    if p.run_this:
        return p



def ensure_global_reference_raster(p):
    """Create a clean global 1 km WGS84 reference raster if needed."""
    reference_path = os.path.join(
        p.base_data_dir,
        'preprocess_data',
        'reference',
        'global_1km_reference.tif',
    )

    if os.path.exists(reference_path):
        return reference_path

    os.makedirs(os.path.dirname(reference_path), exist_ok=True)

    n_cols = 43200
    n_rows = 20880
    pixel_size = 1.0 / 120.0

    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(
        reference_path,
        n_cols,
        n_rows,
        1,
        gdal.GDT_Float32,
        options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'],
    )
    ds.SetGeoTransform((-180.0, pixel_size, 0.0, 84.0, 0.0, -pixel_size))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())

    band = ds.GetRasterBand(1)
    band.Fill(-9999.0)
    band.SetNoDataValue(-9999.0)
    band.FlushCache()
    ds = None

    return reference_path


def preprocess_uglc(p):
    """
    Preprocess the Unified Global Landslide Catalogue (UGLC) into annual
    pixel-level landslide occurrence and mortality panels.

    Source
    ------
    Mancino, S., Sblano, A., Lovergine, F. P., Sethi, T., Capolongo, D., &
    Amatulli, G. (2025). Unified Global Landslide Catalogue (UGLC) [Data set].
    Zenodo. https://doi.org/10.5281/zenodo.18643456

    Notes
    -----
    - Keeps trigger-related fields (TYPE, PHYSICAL FACTORS, RECORD TYPE, etc.)
      in the saved vector for later filtering or regression use.
    - Uses start_date as the default annual anchor via p.uglc_year_source.
      Set p.uglc_year_source = 'end_date' if you want to switch the anchor.
    - Keeps downstream raster filenames the same as the prior GLC pipeline so
      the rest of the workflow does not need to change immediately.
    """
    if not p.run_this:
        return p

    output_dir = os.path.join(*p.base_data, 'preprocess_data', 'uglc')
    binary_out_paths = [os.path.join(output_dir, f'uglc_binary_{year}.tif') for year in p.time_range]
    mortality_out_paths = [os.path.join(output_dir, f'uglc_mortality_{year}.tif') for year in p.time_range]
    out_paths = binary_out_paths + mortality_out_paths
    if all(s3_handler.file_exists(path) for path in out_paths):
        p.L.info(f'All UGLC binary and mortality rasters already exist in {output_dir}/')
        return p

    p.L.info(f'Generating UGLC binary and mortality rasters in {output_dir}/')

    raw_csv_candidates = [
        os.path.join(p.base_data_dir, 'uglc', 'UGLC_point.csv'),
        os.path.join(*p.base_data, 'uglc', 'UGLC_point.csv'),
    ]
    raw_csv_path = None
    for candidate in raw_csv_candidates:
        if os.path.exists(candidate):
            raw_csv_path = candidate
            break
    if raw_csv_path is None:
        raw_csv_path = s3_handler.get_or_cache_local_path(
            os.path.join(*p.base_data, 'uglc', 'UGLC_point.csv'),
            os.path.join(p.project_dir, 'cache', 'uglc'))

    ugcl_df = pd.read_csv(raw_csv_path, sep='|', low_memory=False)

    rename_map = {
        'WKT_GEOM': 'geometry_wkt',
        'NEW DATASET': 'new_dataset',
        'ID': 'uglc_id',
        'OLD DATASET': 'old_dataset',
        'OLD ID': 'old_id',
        'VERSION': 'version',
        'COUNTRY': 'country',
        'ACCURACY': 'accuracy_m',
        'START DATE': 'start_date',
        'END DATE': 'end_date',
        'TYPE': 'landslide_type',
        'PHYSICAL FACTORS': 'physical_factors',
        'RELIABILITY': 'reliability',
        'RECORD TYPE': 'record_type',
        'FATALITIES': 'fatality_count',
        'INJURIES': 'injury_count',
        'NOTES': 'notes',
        'LINK': 'link',
    }
    ugcl_df = ugcl_df.rename(columns={k: v for k, v in rename_map.items() if k in ugcl_df.columns})

    if 'geometry_wkt' not in ugcl_df.columns:
        raise KeyError('UGLC CSV does not contain a WKT_GEOM column.')

    ugcl_gdf = gpd.GeoDataFrame(
        ugcl_df,
        geometry=gpd.GeoSeries.from_wkt(ugcl_df['geometry_wkt']),
        crs='EPSG:4326',
    )

    for date_col in ['start_date', 'end_date']:
        if date_col in ugcl_gdf.columns:
            ugcl_gdf[date_col] = pd.to_datetime(ugcl_gdf[date_col], errors='coerce')

    year_source = getattr(p, 'uglc_year_source', 'start_date')
    if year_source not in {'start_date', 'end_date'}:
        p.L.warning(f'Unsupported uglc_year_source={year_source!r}; defaulting to start_date')
        year_source = 'start_date'
    if year_source not in ugcl_gdf.columns:
        raise KeyError(f'UGLC CSV does not contain required date column: {year_source}')
    ugcl_gdf['event_year'] = ugcl_gdf[year_source].dt.year
    if 'end_date' in ugcl_gdf.columns:
        ugcl_gdf['event_year'] = ugcl_gdf['event_year'].fillna(ugcl_gdf['end_date'].dt.year)

    if 'fatality_count' in ugcl_gdf.columns:
        fatality_series = ugcl_gdf['fatality_count']
    else:
        fatality_series = pd.Series(0, index=ugcl_gdf.index)
    ugcl_gdf['fatality_count'] = pd.to_numeric(fatality_series, errors='coerce').fillna(0).clip(lower=0)

    if 'injury_count' in ugcl_gdf.columns:
        ugcl_gdf['injury_count'] = pd.to_numeric(ugcl_gdf['injury_count'], errors='coerce').fillna(0).clip(lower=0)

    if 'accuracy_m' not in ugcl_gdf.columns:
        raise KeyError('UGLC CSV does not contain an ACCURACY column.')
    ugcl_gdf['accuracy_m'] = pd.to_numeric(ugcl_gdf['accuracy_m'], errors='coerce')
    ugcl_gdf = ugcl_gdf[ugcl_gdf['accuracy_m'].notna()]
    ugcl_gdf = ugcl_gdf[ugcl_gdf['accuracy_m'] <= p.max_location_accuracy_m]

    p.L.info(
        f'UGLC records loaded: {len(ugcl_gdf):,} after accuracy filter; '
        f'using {year_source} as the annual anchor.'
    )

    ugcl_gdf_3857 = ugcl_gdf.to_crs('EPSG:3857')
    ugcl_gdf_3857['center_geom'] = ugcl_gdf_3857['geometry']
    ugcl_gdf_3857['geometry'] = ugcl_gdf_3857.buffer(ugcl_gdf_3857['accuracy_m'])

    to_3857 = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)

    for year in p.time_range:
        binary_out_path = binary_out_paths[p.time_range.index(year)]
        mortality_out_path = mortality_out_paths[p.time_range.index(year)]

        if s3_handler.file_exists(binary_out_path) and s3_handler.file_exists(mortality_out_path):
            p.L.info(f'UGLC binary and mortality rasters for {year} already exist, skipping.')
            continue

        p.L.info(f'Creating UGLC binary and mortality rasters for {year} at {binary_out_path} and {mortality_out_path}.')

        with s3_handler.temp_workspace('uglc') as temp_dir:
            shape = (p.reference_raster_info['raster_size'][1], p.reference_raster_info['raster_size'][0])
            binary_arr = np.full(shape, 0, dtype=np.uint8)
            mortality_arr = np.full(shape, 0.0, dtype=np.float32)
            local_binary_path = os.path.join(temp_dir, f'uglc_binary_{year}.tif')
            local_mortality_path = os.path.join(temp_dir, f'uglc_mortality_{year}.tif')

            pygeo.numpy_array_to_raster(
                binary_arr,
                -1,
                p.reference_raster_info['pixel_size'],
                (p.reference_raster_info['geotransform'][0], p.reference_raster_info['geotransform'][3]),
                p.reference_raster_info['projection_wkt'],
                local_binary_path,
            )
            pygeo.numpy_array_to_raster(
                mortality_arr,
                -9999.0,
                p.reference_raster_info['pixel_size'],
                (p.reference_raster_info['geotransform'][0], p.reference_raster_info['geotransform'][3]),
                p.reference_raster_info['projection_wkt'],
                local_mortality_path,
            )

            yearly_gdf_3857 = ugcl_gdf_3857[ugcl_gdf_3857['event_year'] == year]
            if yearly_gdf_3857.empty:
                p.L.warning(f'No UGLC records found for year {year}, creating empty rasters.')
                s3_handler.upload_from_temp(binary_out_path, filename=f'uglc_binary_{year}.tif')
                s3_handler.upload_from_temp(mortality_out_path, filename=f'uglc_mortality_{year}.tif')
                continue

            yearly_gdf_4326 = yearly_gdf_3857.to_crs('EPSG:4326')
            yearly_gdf_4326 = yearly_gdf_4326.drop(columns=['center_geom'])
            vector_path = os.path.join(temp_dir, f'yearly_{year}.gpkg')
            yearly_gdf_4326.to_file(vector_path, driver='GPKG')
            pygeo.rasterize(
                vector_path,
                local_binary_path,
                burn_values=[1],
                option_list=['ALL_TOUCHED=TRUE'],
                layer_id=0,
            )
            s3_handler.upload_from_temp(binary_out_path, filename=f'uglc_binary_{year}.tif')

            arr = pygeo.raster_to_numpy_array(local_mortality_path)
            info = pygeo.get_raster_info(local_mortality_path)
            transform = info['geotransform']

            def xy_to_rowcol(transform, x, y):
                col = int((x - transform[0]) / transform[1])
                row = int((y - transform[3]) / transform[5])
                return row, col

            event_count = 0
            pixels_updated = 0
            yearly_gdf_4326_loop = ugcl_gdf_3857.to_crs('EPSG:4326')

            for idx, row in ugcl_gdf_3857.iterrows():
                if pd.isna(row['fatality_count']) or row['fatality_count'] <= 0:
                    continue

                event_count += 1
                center_3857 = row['center_geom']
                buffer_radius = row['accuracy_m']
                if pd.isna(buffer_radius) or buffer_radius <= 0:
                    continue

                buffer_geom_4326 = yearly_gdf_4326_loop.loc[idx, 'geometry']
                minx, miny, maxx, maxy = buffer_geom_4326.bounds
                if any(pd.isna(v) for v in [minx, miny, maxx, maxy]):
                    continue

                row_start, col_start = xy_to_rowcol(transform, minx, maxy)
                row_stop, col_stop = xy_to_rowcol(transform, maxx, miny)
                row_start = max(row_start, 0)
                col_start = max(col_start, 0)
                row_stop = min(row_stop, arr.shape[0] - 1)
                col_stop = min(col_stop, arr.shape[1] - 1)

                for r in range(row_start, row_stop + 1):
                    for c in range(col_start, col_stop + 1):
                        x_4326 = transform[0] + c * transform[1]
                        y_4326 = transform[3] + r * transform[5]
                        x_3857, y_3857 = to_3857.transform(x_4326, y_4326)
                        pixel_pt = Point(x_3857, y_3857)
                        dist = center_3857.distance(pixel_pt)
                        if dist <= buffer_radius:
                            weight = 1 - (dist / buffer_radius)
                            arr[r, c] += row['fatality_count'] * weight
                            pixels_updated += 1

            pygeo.numpy_array_to_raster(
                arr,
                -9999.0,
                info['pixel_size'],
                (transform[0], transform[3]),
                info['projection_wkt'],
                local_mortality_path,
                raster_driver_creation_tuple=(
                    'GTIFF',
                    ('TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256'),
                ),
            )
            s3_handler.upload_from_temp(mortality_out_path, filename=f'uglc_mortality_{year}.tif')
            p.L.info(f'{year} mortality raster: {event_count} records with fatalities, {pixels_updated} pixels updated')

    buffered_gpkg_path = os.path.join(*p.base_data, 'preprocess_data', 'uglc', 'uglc_buffered_points.gpkg')
    with s3_handler.temp_workspace('uglc') as temp_dir:
        local_gpkg_path = os.path.join(temp_dir, 'uglc_buffered_points.gpkg')
        ugcl_gdf_to_save = ugcl_gdf_3857.drop(columns=['center_geom'])
        ugcl_gdf_to_save.to_file(local_gpkg_path, driver='GPKG')
        s3_handler.upload_from_temp(buffered_gpkg_path, filename='uglc_buffered_points.gpkg')
    p.L.info(f'Saved buffered landslide points to {buffered_gpkg_path}')

def preprocess_gedtm(p):
    """
    Preprocess GEDTM30 terrain layers to create time-invariant rasters aligned
    to the project reference grid.

    Layers processed
    ----------------
    ls_factor        LS-Factor (slope length × steepness erosion index)
                     raw UInt16, scale 1000  → divide by 1000 → float32
                     source: https://zenodo.org/records/18702591
                     citation: Ho & Hengl (2026), doi:10.5281/zenodo.18702591

    slope_in_degree  Slope in degrees
                     raw UInt16, scale 100   → divide by 100  → float32
                     source: https://zenodo.org/records/18702594
                     citation: Ho & Hengl (2026), doi:10.5281/zenodo.18702594

    twi              Topographic Wetness Index
                     raw Int16,  scale 100   → divide by 100  → float32
                     source: https://zenodo.org/records/18702598
                     citation: Ho & Hengl (2026), doi:10.5281/zenodo.18702598

    Steps
    -----
    For each layer:
        1. Download raw 960 m GeoTIFF from S3
        2. Warp / align to project reference raster (bilinear resampling)
        3. Apply scale-factor division and cast to float32
        4. Write nodata as np.nan (float32 NaN)
        5. Upload processed raster to S3 output location

    Inputs  (S3)
    ------------
    {base_data}/gedtm/ls.factor_gedtm_m_960m_s_20060101_20151231_go_epsg.4326_v1.2.0.tif
    {base_data}/gedtm/slope.in.degree_gedtm_m_960m_s_20060101_20151231_go_epsg.4326_v1.2.0.tif
    {base_data}/gedtm/twi_gedtm_m_960m_s_20060101_20151231_go_epsg.4326_v1.2.0.tif

    Outputs (S3)
    ------------
    {base_data}/preprocess_data/gedtm/lsfactor_gedtm.tif
    {base_data}/preprocess_data/gedtm/slope_degree_gedtm.tif
    {base_data}/preprocess_data/gedtm/twi_gedtm.tif
    """
    if not p.run_this:
        return p

    # ------------------------------------------------------------------
    # Layer registry
    # Each entry: (raw_filename, out_filename, scale_divisor, raw_nodata)
    # raw_nodata values from GEDTM30 documentation:
    #   UInt16 layers (ls_factor, slope): nodata = 65535
    #   Int16  layers (twi):             nodata = 32767
    # ------------------------------------------------------------------
    layers = [
        (
            'ls.factor_gedtm_m_960m_s_20060101_20151231_go_epsg.4326_v1.2.0.tif',
            'lsfactor_gedtm.tif',
            1000.0,
            65535,
        ),
        (
            'slope.in.degree_gedtm_m_960m_s_20060101_20151231_go_epsg.4326_v1.2.0.tif',
            'slope_degree_gedtm.tif',
            100.0,
            65535,
        ),
        (
            'twi_gedtm_m_960m_s_20060101_20151231_go_epsg.4326_v1.2.0.tif',
            'twi_gedtm.tif',
            100.0,
            32767,
        ),
    ]

    all_exist = True
    for _, out_filename, _, _ in layers:
        s3_out = os.path.join(*p.base_data, 'preprocess_data', 'gedtm', out_filename)
        if not s3_handler.file_exists(s3_out):
            all_exist = False
            break

    if all_exist:
        hb.log('All GEDTM layers already exist on S3, skipping.')
        return p

    for raw_filename, out_filename, scale_divisor, raw_nodata in layers:

        s3_raw = os.path.join(*p.base_data, 'gedtm', raw_filename)
        s3_out = os.path.join(*p.base_data, 'preprocess_data', 'gedtm', out_filename)

        if s3_handler.file_exists(s3_out):
            hb.log(f'Already exists, skipping: {s3_out}')
            continue

        hb.log(f'Processing {raw_filename}  (scale ÷{scale_divisor})...')

        with s3_handler.temp_workspace(f'gedtm_{out_filename}') as temp_dir:

            # 1. Download raw raster
            local_raw = s3_handler.download_to_temp(s3_raw)
            local_warped = os.path.join(temp_dir, 'warped.tif')
            local_out    = os.path.join(temp_dir, out_filename)

            # 2. Warp to reference grid (still integer at this stage)
            hb.warp_raster_to_match(
                input_path=str(local_raw),
                output_path=local_warped,
                match_path=p.reference_raster_path,
                resample_method='bilinear',
            )

            # 3. Apply scale factor, set nodata pixels to NaN, save float32
            ds_in = gdal.Open(local_warped, gdal.GA_ReadOnly)
            band  = ds_in.GetRasterBand(1)
            arr   = band.ReadAsArray().astype(np.float32)

            # Mask documented nodata value and any out-of-range fill
            nodata_mask = (arr == np.float32(raw_nodata))
            arr[nodata_mask] = np.nan

            # Apply scale correction
            arr /= np.float32(scale_divisor)

            # Write output as float32 GeoTIFF with NaN nodata
            driver   = gdal.GetDriverByName('GTiff')
            n_cols   = ds_in.RasterXSize
            n_rows   = ds_in.RasterYSize
            ds_out   = driver.Create(
                local_out, n_cols, n_rows, 1, gdal.GDT_Float32,
                options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'],
            )
            ds_out.SetGeoTransform(ds_in.GetGeoTransform())
            ds_out.SetProjection(ds_in.GetProjection())
            band_out = ds_out.GetRasterBand(1)
            band_out.SetNoDataValue(float('nan'))
            band_out.WriteArray(arr)
            band_out.FlushCache()
            ds_out = ds_in = None          # close both datasets

            # 4. Upload to S3
            s3_handler.upload_from_temp(s3_out, local_out)
            hb.log(f'Saved: {s3_out}')

    hb.log('GEDTM preprocessing complete.')
    return p

def preprocess_geomorpho90m(p):
    """
    Preprocess Geomorpho90m terrain layers to create time-invariant rasters aligned
    to the project reference grid.

    Layers processed
    ----------------
    slope              Slope in degrees (MERIT-DEM, replaces/supplements GEDTM slope)
                       raw Int16, scale 0.01 → float32 degrees
                       source: https://www.geomorpho90m.org/
                       citation: Amatulli et al. (2020), doi:10.1038/s41597-020-0479-6

    roughness          Terrain ruggedness index (local elevation range)
                       raw Int16, scale 0.1 → float32
                       source: https://www.geomorpho90m.org/

    tpi                Topographic Position Index (ridge/valley/slope classification)
                       raw Int16, scale 0.1 → float32
                       source: https://www.geomorpho90m.org/

    elev_stdev         Standard deviation of elevation (local terrain variability)
                       raw Int16, scale 0.1 → float32
                       source: https://www.geomorpho90m.org/

    aspect             Aspect in degrees (0–360, where 0=N, 90=E, 180=S, 270=W)
                       raw UInt16, scale 0.01 → float32 degrees
                       source: https://www.geomorpho90m.org/

    Steps
    -----
    For each layer:
        1. Download raw 250 m GeoTIFF from S3
        2. Inspect source raster for actual nodata value
        3. Warp / align to project reference raster (bilinear resampling)
        4. Clean output: mask any values outside valid ranges as NaN, ensure all nodata → NaN
        5. Write as float32 with NaN nodata
        6. Upload processed raster to S3 output location

    Inputs  (S3)
    ------------
    {base_data}/geomorpho90m/dtm_slope_merit.dem_m_250m_s0..0cm_2018_v1.0.tif
    {base_data}/geomorpho90m/dtm_roughness_merit.dem_m_250m_s0..0cm_2018_v1.0.tif
    {base_data}/geomorpho90m/dtm_tpi_merit.dem_m_250m_s0..0cm_2018_v1.0.tif
    {base_data}/geomorpho90m/dtm_elev-stdev_merit.dem_m_250m_s0..0cm_2018_v1.0.tif
    {base_data}/geomorpho90m/dtm_aspect_merit.dem_m_250m_s0..0cm_2018_v1.0.tif

    Outputs (S3)
    ------------
    {base_data}/preprocess_data/geomorpho90m/slope_geomorpho.tif
    {base_data}/preprocess_data/geomorpho90m/roughness_geomorpho.tif
    {base_data}/preprocess_data/geomorpho90m/tpi_geomorpho.tif
    {base_data}/preprocess_data/geomorpho90m/elev_stdev_geomorpho.tif
    {base_data}/preprocess_data/geomorpho90m/aspect_geomorpho.tif
    """
    if not p.run_this:
        return p

    # ------------------------------------------------------------------
        # Layer registry with scaling and validity ranges.
        # Each entry: (raw_filename, out_filename, scale_factor, offset, min_valid, max_valid)
        # Validity bounds are in physical units after scaling.
    # ------------------------------------------------------------------
    layers = [
        ('dtm_slope_merit.dem_m_250m_s0..0cm_2018_v1.0.tif',
         'slope_geomorpho.tif',
         0.01, 0.0, 0, 90),            # slope: 0-90 degrees
        ('dtm_roughness_merit.dem_m_250m_s0..0cm_2018_v1.0.tif',
         'roughness_geomorpho.tif',
         0.1, 0.0, 0, None),          # roughness: ≥0 (no upper bound)
        ('dtm_tpi_merit.dem_m_250m_s0..0cm_2018_v1.0.tif',
         'tpi_geomorpho.tif',
         0.1, 0.0, None, None),       # TPI: can be negative (valley) or positive (ridge)
        ('dtm_elev-stdev_merit.dem_m_250m_s0..0cm_2018_v1.0.tif',
         'elev_stdev_geomorpho.tif',
         0.1, 0.0, 0, None),          # elev_stdev: ≥0 (no upper bound)
        ('dtm_aspect_merit.dem_m_250m_s0..0cm_2018_v1.0.tif',
         'aspect_geomorpho.tif',
         0.01, 0.0, 0, 360),          # aspect: 0-360 degrees
    ]

    for raw_filename, out_filename, scale_factor, offset, min_valid, max_valid in layers:

        s3_raw = os.path.join(*p.base_data, 'geomorpho90m', raw_filename)
        s3_out = os.path.join(*p.base_data, 'preprocess_data', 'geomorpho90m', out_filename)

        if s3_handler.file_exists(s3_out):
            hb.log(f'Already exists, skipping: {s3_out}')
            continue

        hb.log(f'Processing {raw_filename} ...')

        with s3_handler.temp_workspace(f'geomorpho90m_{out_filename}') as temp_dir:

            # 1. Download raw raster and inspect its nodata value
            local_raw = s3_handler.download_to_temp(s3_raw)
            
            # Read source metadata to get actual nodata value
            ds_src = gdal.Open(str(local_raw), gdal.GA_ReadOnly)
            if ds_src is None:
                raise RuntimeError(f'Failed to open source raster: {local_raw}')
            band_src = ds_src.GetRasterBand(1)
            src_nodata = band_src.GetNoDataValue()
            src_scale = band_src.GetScale()
            src_offset = band_src.GetOffset()
            if src_scale is None:
                src_scale = scale_factor
            if src_offset is None:
                src_offset = offset
            hb.log(f'  Source nodata value: {src_nodata}')
            hb.log(f'  Source scale/offset: {src_scale} / {src_offset}')
            ds_src = None
            
            local_warped = os.path.join(temp_dir, 'warped.tif')
            local_out = os.path.join(temp_dir, out_filename)

            # 2. Warp to reference grid (bilinear resampling for continuous terrain data)
            # Note: src_nodata may be None, which is fine - GDAL will handle it
            hb.warp_raster_to_match(
                input_path=str(local_raw),
                output_path=local_warped,
                match_path=p.reference_raster_path,
                resample_method='bilinear',
                src_ndv=src_nodata,
                dst_ndv=src_nodata,
            )

            # 3. Read warped raster and aggressively clean it
            ds_in = gdal.Open(local_warped, gdal.GA_ReadOnly)
            if ds_in is None:
                raise RuntimeError(f'Failed to open warped raster: {local_warped}')
            band_in = ds_in.GetRasterBand(1)
            raw_arr = band_in.ReadAsArray().astype(np.float32)
            arr = (raw_arr * np.float32(src_scale)) + np.float32(src_offset)
            
            # Get warped raster's nodata value
            warped_nodata = band_in.GetNoDataValue()
            hb.log(f'  Warped nodata value: {warped_nodata}')
            
            # Build mask for all invalid values
            invalid_mask = np.zeros(arr.shape, dtype=bool)
            
            # Mask source nodata if it exists and is finite
            if src_nodata is not None and np.isfinite(float(src_nodata)):
                invalid_mask |= (raw_arr == np.float32(src_nodata))
                hb.log(f'    Masking source nodata {src_nodata}')
            
            # Mask warped nodata if it exists and is finite
            if warped_nodata is not None and np.isfinite(float(warped_nodata)):
                invalid_mask |= (raw_arr == np.float32(warped_nodata))
                hb.log(f'    Masking warped nodata {warped_nodata}')
            
            # Mask any NaN or infinite values
            invalid_mask |= ~np.isfinite(arr)
            
            # Apply domain-specific validity range checks
            if min_valid is not None:
                invalid_mask |= (arr < np.float32(min_valid))
                hb.log(f'    Masking values < {min_valid}')
            if max_valid is not None:
                invalid_mask |= (arr > np.float32(max_valid))
                hb.log(f'    Masking values > {max_valid}')
            
            # Diagnostic: count masked pixels
            n_masked = np.sum(invalid_mask)
            n_total = arr.size
            hb.log(f'    Masked {n_masked:,} / {n_total:,} pixels ({100*n_masked/n_total:.2f}%)')
            
            # Set all invalid pixels to NaN
            arr[invalid_mask] = np.nan

            # 4. Write cleaned float32 raster with NaN nodata
            driver = gdal.GetDriverByName('GTiff')
            ds_out = driver.Create(
                local_out,
                ds_in.RasterXSize,
                ds_in.RasterYSize,
                1,
                gdal.GDT_Float32,
                options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'],
            )
            ds_out.SetGeoTransform(ds_in.GetGeoTransform())
            ds_out.SetProjection(ds_in.GetProjection())
            band_out = ds_out.GetRasterBand(1)
            band_out.SetNoDataValue(float('nan'))
            band_out.WriteArray(arr)
            band_out.FlushCache()
            ds_out = ds_in = None

            # 5. Upload to S3
            s3_handler.upload_from_temp(s3_out, local_out)
            hb.log(f'Saved: {s3_out}')

    hb.log('Geomorpho90m preprocessing complete.')
    return p


def preprocess_gaez(p):
    """
    Preprocess GAEZ data to create a time-invariant raster of global agro-ecological suitability.
    1. Load raw GAEZ data
        format: GeoTIFF raster
        source: 
        citation: 
    2. Reproject and align with reference
    """
    if p.run_this:
        s3_out_path = os.path.join(*p.base_data, 'preprocess_data', 'fao_gaez', 'fao_gaez.tif')
        if s3_handler.file_exists(s3_out_path):
            p.L.info(f'GAEZ raster already exists at {s3_out_path}')
            return p
        else:
            p.L.info(f'Generating GAEZ raster at {s3_out_path}')
        raw_path = os.path.join(*p.base_data, 'fao_gaez', 'GAEZ-V5.AEZ57.tif')
        with s3_handler.temp_workspace('fao_gaez') as temp_dir:
            local_raw_path = s3_handler.download_to_temp(raw_path)
            local_out_path = os.path.join(temp_dir, 'fao_gaez.tif')
            hb.warp_raster_to_match(
                input_path=str(local_raw_path),
                output_path=local_out_path,
                match_path=p.reference_raster_path,
                resample_method='nearest',
                src_ndv=255,
                dst_ndv=0,
                output_data_type=gdal.GDT_Byte,
            )
            # Move result to final S3 location
            s3_handler.upload_from_temp(s3_out_path, local_out_path)


def preprocess_travel_time(p):
    """
    Preprocess travel-time-to-healthcare rasters (motorized and walking-only).
    
    These are time-invariant (2020) global 1 km resolution rasters measuring
    travel time (in hours) to the nearest healthcare facility.
    
    Sources
    -------
    Weiss, D.J., Nelson, A., Vargas-Ruiz, C.A. et al. (2020).
    Global maps of travel time to healthcare facilities. Nat Med 26, 1835–1838.
    https://doi.org/10.1038/s41591-020-1059-1
    
    Inputs (S3)
    -----------
    {base_data}/malariaatlas/2020_motorized_travel_time_to_healthcare.geotiff
    {base_data}/malariaatlas/2020_walking_only_travel_time_to_healthcare.geotiff
    
    Outputs (S3)
    -----------
    {base_data}/preprocess_data/malariaatlas/motorized_travel_time_to_healthcare.tif
    {base_data}/preprocess_data/malariaatlas/walking_only_travel_time_to_healthcare.tif
    """
    if not p.run_this:
        return p

    # Layer registry: (input_filename, output_filename)
    layers = [
        ('2020_motorized_travel_time_to_healthcare.geotiff', 'motorized_travel_time_to_healthcare.tif'),
        ('2020_walking_only_travel_time_to_healthcare.geotiff', 'walking_only_travel_time_to_healthcare.tif'),
    ]

    # Check if all outputs exist
    all_exist = True
    for _, out_filename in layers:
        s3_out = os.path.join(*p.base_data, 'preprocess_data', 'travel_time', out_filename)
        if not s3_handler.file_exists(s3_out):
            all_exist = False
            break

    if all_exist:
        hb.log('All travel-time rasters already exist on S3, skipping.')
        return p

    for raw_filename, out_filename in layers:
        s3_raw = os.path.join(*p.base_data, 'malariaatlas', raw_filename)
        s3_out = os.path.join(*p.base_data, 'preprocess_data', 'malariaatlas', out_filename)

        if s3_handler.file_exists(s3_out):
            hb.log(f'Already exists, skipping: {s3_out}')
            continue

        if not s3_handler.file_exists(s3_raw):
            hb.log(f'WARNING: input not found, skipping: {s3_raw}')
            continue

        hb.log(f'Processing {raw_filename}...')

        with s3_handler.temp_workspace(f'travel_time_{out_filename}') as temp_dir:

            # 1. Download raw raster
            local_raw = s3_handler.download_to_temp(s3_raw)
            local_warped = os.path.join(temp_dir, 'warped.tif')
            local_out    = os.path.join(temp_dir, out_filename)

            # 2. Warp to reference grid (bilinear resampling preserves continuous travel time)
            hb.warp_raster_to_match(
                input_path=str(local_raw),
                output_path=local_warped,
                match_path=p.reference_raster_path,
                resample_method='bilinear',
                src_ndv=-9999.0,
                dst_ndv=-9999.0,
            )

            # 3. Upload to S3
            s3_handler.upload_from_temp(s3_out, local_warped)
            hb.log(f'Saved: {s3_out}')

    hb.log('Travel-time preprocessing complete.')
    return p


def preprocess_landscan(p):
    """
    Preprocess LandScan Global annual population data to create aligned population rasters.
    
    1. Load raw LandScan data
        format: GeoTIFF raster (1km resolution, WGS84)
        source: https://landscan.ornl.gov/
        citation: Rose, A., McKee, J., Sims, K., Bright, E., Reith, A., & Urban, M. (2020). 
                  LandScan Global 2019 [Data set]. Oak Ridge National Laboratory. 
                  https://doi.org/10.48690/1524214
        advantage: Annual coverage (no gaps like GPW's 5-year intervals)
    
    2. For each year, reproject and align with reference grid
        - Bilinear resampling (preserves continuous population density)
        - Handle nodata sentinel values
        - Cast to float32
    """
    if not getattr(p, 'run_this', True):
        return p

    # S3 input base
    s3_annual_base = os.path.join(*p.base_data, 'landscan')
    # Output base
    out_base = os.path.join(*p.base_data, 'preprocess_data', 'landscan')

    # Build all output paths
    all_out_paths = []
    for year in p.time_range:
        out_path = os.path.join(out_base, f'landscan_{year}_1km.tif')
        all_out_paths.append(out_path)
    # Check if all outputs exist
    if all(s3_handler.file_exists(path) for path in all_out_paths):
        p.L.info('All warped LandScan annual rasters already exist, skipping.')
        return p

    for year in p.time_range:
        out_path = os.path.join(out_base, f'landscan_{year}_1km.tif')
        if s3_handler.file_exists(out_path):
            p.L.info(f'{year}: output already exists, skipping.')
            continue
        # S3 input path
        s3_in_path = os.path.join(s3_annual_base, f'landscan-global-{year}.tif')
        if not s3_handler.file_exists(s3_in_path):
            p.L.warning(f'{year}: input not found on S3: {s3_in_path}')
            continue
        p.L.info(f'{year}: warping {s3_in_path} to {out_path}')
        with s3_handler.temp_workspace(f'landscan_{year}') as temp_dir:
            local_in = os.path.join(temp_dir, f'in_landscan_{year}.tif')
            local_out = os.path.join(temp_dir, f'warped_landscan_{year}.tif')
            # Download from S3
            s3_handler.download_to_temp(s3_in_path, local_in)

            # Warp to reference
            hb.warp_raster_to_match(
                input_path=local_in,
                output_path=local_out,
                match_path=p.reference_raster_path,
                resample_method='bilinear',
                src_ndv=-2147483648,
                dst_ndv=-9999.0,
                output_data_type=gdal.GDT_Float32,
            )

            # Upload to S3
            s3_handler.upload_from_temp(out_path, local_out)
            p.L.info(f'{year}: warped and uploaded to {out_path}')
    p.L.info('LandScan annual warping complete.')
    return p


# NOTE: Moved to MSI, download times too slow
# def preprocess_era5(p):
#     """
#     Preprocess ERA5-Land monthly precipitation data to compute annual extreme rainfall metrics:
#     1. Annual maximum daily rainfall (mm)
#     2. Annual maximum 3-day cumulative rainfall (mm) — proxy for antecedent wetness
#     3. Annual maximum hourly intensity (mm/hr) — proxy for flash triggering

#     Processes month-by-month to avoid loading >2GB files simultaneously.
#     Handles 3-day rolling window across month boundaries by carrying last 2 days forward.
#     Outputs are GeoTIFFs aligned to p.reference_raster_path.
#     """
#     if p.run_this:

#         # Check if all outputs already exist
#         all_out_paths = (
#             [os.path.join(*p.base_data, 'preprocess_data', 'era5_land', f'era5_max_daily_mm_{year}.tif')  for year in p.time_range] +
#             [os.path.join(*p.base_data, 'preprocess_data', 'era5_land', f'era5_max_3day_mm_{year}.tif')   for year in p.time_range] +
#             [os.path.join(*p.base_data, 'preprocess_data', 'era5_land', f'era5_max_hourly_mm_{year}.tif') for year in p.time_range]
#         )
#         if all(s3_handler.file_exists(path) for path in all_out_paths):
#             p.L.info('All ERA5 output rasters already exist, skipping.')
#             return p

#         # ------------------------------------------------------------------ #
#         # Main loop: process year by year, month by month                     #
#         # ------------------------------------------------------------------ #
#         for year in p.time_range:
#             out_max_daily_s3  = os.path.join(*p.base_data, 'preprocess_data', 'era5_land', f'era5_max_daily_mm_{year}.tif')
#             out_max_3day_s3   = os.path.join(*p.base_data, 'preprocess_data', 'era5_land', f'era5_max_3day_mm_{year}.tif')
#             out_max_hourly_s3 = os.path.join(*p.base_data, 'preprocess_data', 'era5_land', f'era5_max_hourly_mm_{year}.tif')

#             if (s3_handler.file_exists(out_max_daily_s3) and
#                     s3_handler.file_exists(out_max_3day_s3) and
#                     s3_handler.file_exists(out_max_hourly_s3)):
#                 p.L.info(f'{year}: all outputs already exist, skipping.')
#                 continue

#             p.L.info(f'{year}: starting processing...')

#             # Accumulators — shape set on first month
#             annual_max_daily  = None
#             annual_max_3day   = None
#             annual_max_hourly = None
#             carry_days        = None  # last 2 days of previous month for boundary rolling

#             with s3_handler.temp_workspace(f'era5_{year}') as temp_dir:
#                 lat_vals = None
#                 lon_vals = None
#                 for month in range(1, 13):
#                 # for month in [1]:  # Only process January for testing
#                     s3_nc_path = os.path.join(
#                         *p.base_data, 'era5_land',
#                         f'era5_land_precip_{year}_{month:02d}.nc'
#                     )
#                     if not s3_handler.file_exists(s3_nc_path):
#                         raise FileNotFoundError(f'Missing monthly file: {s3_nc_path}')

#                     local_nc_path = os.path.join(temp_dir, f'era5_land_precip_{year}_{month:02d}.nc')
#                     p.L.info(f'{year}-{month:02d}: downloading...')
#                     s3_handler.download_to_temp(s3_nc_path, local_nc_path)

#                     p.L.info(f'--- {year}-{month:02d} ---')

#                     # Check if file is a ZIP (even if .nc extension)
#                     nc_to_open = local_nc_path
#                     try:
#                         with zipfile.ZipFile(local_nc_path, 'r') as zf:
#                             nc_files = [f for f in zf.namelist() if f.endswith('.nc')]
#                             if nc_files:
#                                 zf.extract(nc_files[0], temp_dir)
#                                 nc_to_open = os.path.join(temp_dir, nc_files[0])
#                                 p.L.info(f"Extracted NetCDF from ZIP: {nc_to_open}")
#                     except zipfile.BadZipFile:
#                         # Not a zip, proceed as normal
#                         pass
#                     # Use netcdf4 engine explicitly to avoid backend ambiguity error
#                     p.L.info(f'{year}-{month:02d}: opening NetCDF file...')
#                     ds = xr.open_dataset(nc_to_open, engine='netcdf4', chunks={'time': 24})

#                     precip = ds['tp'] if 'tp' in ds else ds['total_precipitation']

#                     # Fix longitude
#                     if float(precip.longitude.max()) > 180:
#                         precip = precip.assign_coords(
#                             longitude=(((precip.longitude + 180) % 360) - 180)
#                         )
#                         precip = precip.sortby('longitude')

#                     # Capture ERA5 grid coords from the first month, before ds.close()
#                     if lat_vals is None:
#                         lat_vals = precip.latitude.values.copy()
#                         lon_vals = precip.longitude.values.copy()

#                     precip_mm = precip * 1000
#                     precip_hourly = precip_mm.diff(dim='valid_time').clip(min=0).fillna(0)

#                     daily_mm             = precip_hourly.resample(valid_time='1D').sum()
#                     daily_max_hourly     = precip_hourly.resample(valid_time='1D').max()
#                     daily_vals           = daily_mm.compute().values
#                     daily_max_hourly_vals = daily_max_hourly.compute().values

#                     ds.close()   # arrays are in numpy

#                     # Initialize accumulators on first month
#                     if annual_max_daily is None:
#                         spatial_shape     = daily_vals.shape[1:]
#                         annual_max_daily  = np.full(spatial_shape, -np.inf)
#                         annual_max_3day   = np.full(spatial_shape, -np.inf)
#                         annual_max_hourly = np.full(spatial_shape, -np.inf)

#                     # Update accumulators
#                     annual_max_daily  = np.fmax(annual_max_daily,  daily_vals.max(axis=0))
#                     annual_max_hourly = np.fmax(annual_max_hourly, daily_max_hourly_vals.max(axis=0))

#                     if carry_days is not None:
#                         extended = np.concatenate([carry_days, daily_vals], axis=0)
#                     else:
#                         extended = daily_vals
#                     n_days = extended.shape[0]
#                     rolling_3day = (extended[0:n_days-2] + extended[1:n_days-1] + extended[2:n_days])
#                     annual_max_3day = np.fmax(annual_max_3day, rolling_3day.max(axis=0))
#                     carry_days = daily_vals[-2:, :, :]

#                     os.remove(local_nc_path)

#                 # --- After month loop: write tifs using ERA5's own grid ---
#                 lon_res = float(lon_vals[1] - lon_vals[0])
#                 lat_res = float(lat_vals[1] - lat_vals[0])   # negative (N->S)
#                 era5_origin      = (float(lon_vals[0]) - lon_res / 2,
#                                     float(lat_vals[0]) - lat_res / 2)
#                 era5_pixel_size  = (lon_res, lat_res)

#                 local_max_daily_tif  = os.path.join(temp_dir, f'era5_max_daily_mm_{year}.tif')
#                 local_max_3day_tif   = os.path.join(temp_dir, f'era5_max_3day_mm_{year}.tif')
#                 local_max_hourly_tif = os.path.join(temp_dir, f'era5_max_hourly_mm_{year}.tif')
#                 warped_daily_tif     = os.path.join(temp_dir, f'era5_max_daily_mm_{year}_warped.tif')
#                 warped_3day_tif      = os.path.join(temp_dir, f'era5_max_3day_mm_{year}_warped.tif')
#                 warped_hourly_tif    = os.path.join(temp_dir, f'era5_max_hourly_mm_{year}_warped.tif')

#                 nodata = -9999.0
#                 annual_max_daily  = np.where(np.isnan(annual_max_daily),  nodata, annual_max_daily).astype(np.float32)
#                 annual_max_3day   = np.where(np.isnan(annual_max_3day),   nodata, annual_max_3day).astype(np.float32)
#                 annual_max_hourly = np.where(np.isnan(annual_max_hourly), nodata, annual_max_hourly).astype(np.float32)

#                 annual_max_daily  = np.where(np.isinf(annual_max_daily),  nodata, annual_max_daily).astype(np.float32)
#                 annual_max_3day   = np.where(np.isinf(annual_max_3day),   nodata, annual_max_3day).astype(np.float32)
#                 annual_max_hourly = np.where(np.isinf(annual_max_hourly), nodata, annual_max_hourly).astype(np.float32)

#                 pygeo.numpy_array_to_raster(annual_max_daily,  nodata, era5_pixel_size, era5_origin, 'EPSG:4326', local_max_daily_tif)
#                 pygeo.numpy_array_to_raster(annual_max_3day,   nodata, era5_pixel_size, era5_origin, 'EPSG:4326', local_max_3day_tif)
#                 pygeo.numpy_array_to_raster(annual_max_hourly, nodata, era5_pixel_size, era5_origin, 'EPSG:4326', local_max_hourly_tif)

#                 hb.warp_raster_to_match(
#                     input_path=local_max_daily_tif,
#                     output_path=warped_daily_tif,
#                     match_path=p.reference_raster_path,
#                     resample_method='bilinear',
#                     src_ndv=-9999.0,
#                     dst_ndv=-9999.0,
#                     output_data_type=gdal.GDT_Float32,
#                 )
#                 hb.warp_raster_to_match(
#                     input_path=local_max_3day_tif,
#                     output_path=warped_3day_tif,
#                     match_path=p.reference_raster_path,
#                     resample_method='bilinear',
#                     src_ndv=-9999.0,
#                     dst_ndv=-9999.0,
#                     output_data_type=gdal.GDT_Float32,
#                 )
#                 hb.warp_raster_to_match(
#                     input_path=local_max_hourly_tif,
#                     output_path=warped_hourly_tif,
#                     match_path=p.reference_raster_path,
#                     resample_method='bilinear',
#                     src_ndv=-9999.0,
#                     dst_ndv=-9999.0,
#                     output_data_type=gdal.GDT_Float32,
#                 )
                
#                 # Upload warped rasters to S3
#                 s3_handler.upload_from_temp(out_max_daily_s3,  warped_daily_tif)
#                 s3_handler.upload_from_temp(out_max_3day_s3,   warped_3day_tif)
#                 s3_handler.upload_from_temp(out_max_hourly_s3, warped_hourly_tif)
#                 p.L.info(f'{year}: uploaded all three warped GeoTIFFs to S3.')

#         p.L.info('ERA5 preprocessing complete.')
#     return p

def preprocess_era5(p):
    """
    Warps annual ERA5 GeoTIFFs from S3 (annual_tif) to the project reference grid, saving to preprocess_data/era5_land.
    Checks for output existence before processing each file. Only runs for years 2000-2020.
    """
    if not getattr(p, 'run_this', True):
        return p

    # Output paths for each metric
    metrics = [
        ('max_daily', 'era5_max_daily_mm'),
        ('max_3day', 'era5_max_3day_mm'),
        ('max_hourly', 'era5_max_hourly_mm'),
    ]
    # S3 input base
    s3_annual_base = os.path.join(*p.base_data, 'era5_land', 'annual_tif')
    # Output base
    out_base = os.path.join(*p.base_data, 'preprocess_data', 'era5_land')

    # Build all output paths
    all_out_paths = []
    for _, out_prefix in metrics:
        for year in p.time_range:
            out_path = os.path.join(out_base, f'{out_prefix}_{year}.tif')
            all_out_paths.append(out_path)
    # Check if all outputs exist
    if all(s3_handler.file_exists(path) for path in all_out_paths):
        p.L.info('All warped ERA5 annual rasters already exist, skipping.')
        return p

    for metric, out_prefix in metrics:
        for year in p.time_range:
            out_path = os.path.join(out_base, f'{out_prefix}_{year}.tif')
            if s3_handler.file_exists(out_path):
                p.L.info(f'{year} {metric}: output already exists, skipping.')
                continue
            # S3 input path
            s3_in_path = os.path.join(s3_annual_base, f'{out_prefix}_{year}.tif')
            if not s3_handler.file_exists(s3_in_path):
                p.L.warning(f'{year} {metric}: input not found on S3: {s3_in_path}')
                continue
            p.L.info(f'{year} {metric}: warping {s3_in_path} to {out_path}')
            with s3_handler.temp_workspace(f'era5warp_{year}_{metric}') as temp_dir:
                local_in = os.path.join(temp_dir, f'in_{out_prefix}_{year}.tif')
                local_out = os.path.join(temp_dir, f'warped_{out_prefix}_{year}.tif')
                # Download from S3
                s3_handler.download_to_temp(s3_in_path, local_in)
                # Warp to reference
                hb.warp_raster_to_match(
                    input_path=local_in,
                    output_path=local_out,
                    match_path=p.reference_raster_path,
                    resample_method='bilinear',
                    src_ndv=-9999.0,
                    dst_ndv=-9999.0,
                    output_data_type=gdal.GDT_Float32,
                )
                # Upload to S3
                s3_handler.upload_from_temp(out_path, local_out)
                p.L.info(f'{year} {metric}: warped and uploaded to {out_path}')
    p.L.info('ERA5 annual warping complete.')
    return p

def preprocess_esacci_to_share(p):
    if not getattr(p, 'run_this', True):
        return p

    correspondence_path = os.path.join(p.base_data_dir, 'esa_seals7_correspondence.csv')
    corr_df = pd.read_csv(correspondence_path)
    # Clean potential duplicate header rows or non-numeric src_id values
    corr_df = corr_df[corr_df['src_id'].apply(lambda x: str(x).strip().isdigit())].copy()
    corr_df['src_id'] = corr_df['src_id'].astype(int)

    # Normalize labels
    corr_df['dst_label'] = corr_df['dst_label'].astype(str).str.strip().str.lower()
    refined_col = 'dst_label_refined'
    if refined_col in corr_df.columns:
        corr_df[refined_col] = corr_df[refined_col].astype(str).str.strip().str.lower()

    # Identify bare_areas src_ids from refined mapping (works for both coarse & refined CF)
    bare_src = set(int(x) for x in corr_df[corr_df[refined_col] == 'bare_areas']['src_id']) if refined_col in corr_df.columns else set()
    p.L.debug(f'Bare areas src_ids: {sorted(list(bare_src))}')

    # Coarse mappings
    categories = list(corr_df['dst_label'].unique())
    category_code_map = {cat: set(int(x) for x in corr_df[corr_df['dst_label'] == cat]['src_id']) for cat in categories}
    noveg_labels = ['forest', 'cropland', 'othernat']
    category_noveg_code_map = {cat: (bare_src.copy() if cat in noveg_labels else set(int(x) for x in corr_df[corr_df['dst_label'] == cat]['src_id'])) for cat in categories}

    # Refined mappings
    refined_categories = []
    refined_code_map = {}
    refined_noveg_code_map = {}
    refined_noveg_labels = ['forest_dense', 'forest_open', 'cropland_rainfed', 'cropland_irrigated', 'grassland', 'shrubland', 'sparse']
    if refined_col in corr_df.columns:
        refined_categories = [x for x in corr_df[refined_col].dropna().unique() if str(x) != 'nan']
        for cat in refined_categories:
            refined_code_map[cat] = set(int(x) for x in corr_df[corr_df[refined_col] == cat]['src_id'])
            refined_noveg_code_map[cat] = (bare_src.copy() if cat in refined_noveg_labels else set(int(x) for x in corr_df[corr_df[refined_col] == cat]['src_id']))

    # Modes: (name, categories, code_map, noveg_map, out_base)
    modes = [
        ('coarse', categories, category_code_map, category_noveg_code_map, os.path.join(*p.base_data, 'preprocess_data', 'esa'))
    ]
    if refined_categories:
        modes.append(('refined', refined_categories, refined_code_map, refined_noveg_code_map, os.path.join(*p.base_data, 'preprocess_data', 'esa_refined')))

    # Quick existence check
    all_exist = True
    for mode_name, cat_list, _, _, out_base in modes:
        for cat in cat_list:
            for year in p.time_range:
                out_name = f'esacci_share_{cat}_{year}.tif' if mode_name == 'coarse' else f'esacci_share_refined_{cat}_{year}.tif'
                if not s3_handler.file_exists(os.path.join(out_base, out_name)):
                    all_exist = False
                    break
            if not all_exist:
                break
        if not all_exist:
            break

    if all_exist:
        p.L.info('All ESA CCI share rasters already exist (coarse + refined).')
        return p

    all_src_ids = sorted(set(int(x) for x in corr_df['src_id']))

    for year in p.time_range:
        raw_path = os.path.join(p.user_dir, 'Files', 'base_data', 'lulc', 'esa', f'lulc_esa_{year}.tif')
        with s3_handler.temp_workspace('esa_share') as temp_dir:
            for mode_name, cat_list, code_map, noveg_map, out_base in modes:
                for cat in cat_list:
                    # skip water in coarse
                    if mode_name == 'coarse' and str(cat) == '6':
                        p.L.info(f'Skipping category 6 (water) for {year}.')
                        continue

                    suffix = f'{cat}_{year}.tif'
                    if mode_name == 'coarse':
                        bin_obs = os.path.join(temp_dir, f'esacci_binary_{suffix}')
                        share_obs = os.path.join(temp_dir, f'esacci_share_{suffix}')
                        s3_share_obs = os.path.join(out_base, f'esacci_share_{suffix}')
                        bin_cf = os.path.join(temp_dir, f'esacci_binary_cf_{suffix}')
                        share_cf = os.path.join(temp_dir, f'esacci_share_cf_{suffix}')
                        s3_share_cf = os.path.join(out_base, f'esacci_share_cf_{suffix}')
                    else:
                        bin_obs = os.path.join(temp_dir, f'esacci_binary_refined_{suffix}')
                        share_obs = os.path.join(temp_dir, f'esacci_share_refined_{suffix}')
                        s3_share_obs = os.path.join(out_base, f'esacci_share_refined_{suffix}')
                        bin_cf = os.path.join(temp_dir, f'esacci_binary_refined_cf_{suffix}')
                        share_cf = os.path.join(temp_dir, f'esacci_share_refined_cf_{suffix}')
                        s3_share_cf = os.path.join(out_base, f'esacci_share_refined_cf_{suffix}')

                    obs_set = code_map.get(cat, set())
                    value_map_obs = {int(src_id): (1 if int(src_id) in obs_set else 0) for src_id in all_src_ids}
                    value_map_obs[255] = 0
                    if not s3_handler.file_exists(s3_share_obs):
                        hb.log(f'[{mode_name}] Observed reclass for {cat} {year}: ones={sum(value_map_obs.values())}')
                        hb.reclassify_raster_hb(input_raster_path=raw_path, rules=value_map_obs, output_raster_path=bin_obs, output_data_type=gdal.GDT_Float32, output_ndv=-9999.0, verbose=True)
                        ds_b = gdal.Open(bin_obs, gdal.GA_ReadOnly)
                        b = ds_b.GetRasterBand(1)
                        stats = b.GetStatistics(False, True)
                        bmin, bmax = float(stats[0]), float(stats[1])
                        if bmin < -1e-6 or bmax > 1.0 + 1e-6:
                            raise ValueError(f'Binary raster values outside [0,1]: min={bmin}, max={bmax} ({bin_obs})')
                        s3_handler.upload_from_temp(os.path.join(out_base, os.path.basename(bin_obs)), bin_obs)
                        hb.warp_raster_to_match(input_path=str(bin_obs), output_path=share_obs, match_path=p.reference_raster_path, resample_method='average', src_ndv=-9999.0, dst_ndv=-9999.0, output_data_type=gdal.GDT_Float32)
                        ds_s = gdal.Open(share_obs, gdal.GA_ReadOnly)
                        bs = ds_s.GetRasterBand(1)
                        stats_s = bs.GetStatistics(False, True)
                        smin, smax = float(stats_s[0]), float(stats_s[1])
                        if smin < -1e-6 or smax > 1.0 + 1e-6:
                            raise ValueError(f'Share raster values outside [0,1]: min={smin}, max={smax} ({share_obs})')
                        else:
                            hb.log(f'[{mode_name}] Observed share raster for {cat} {year} has valid range: min={smin}, max={smax}')
                        s3_handler.upload_from_temp(s3_share_obs, share_obs)
                        p.L.info(f'Created [{mode_name}] observed share raster for {cat} {year} at {s3_share_obs}')

                    cf_set = noveg_map.get(cat, set())
                    value_map_cf = {int(src_id): (1 if int(src_id) in cf_set else 0) for src_id in all_src_ids}
                    value_map_cf[255] = 0
                    p.L.info(f'[{mode_name}] CF mapping for {cat}: noveg_count={len(cf_set)} sample={sorted(list(cf_set))[:10]}')
                    if not s3_handler.file_exists(s3_share_cf):
                        hb.log(f'[{mode_name}] Counterfactual reclass for {cat} {year}: ones={sum(value_map_cf.values())}')
                        hb.reclassify_raster_hb(input_raster_path=raw_path, rules=value_map_cf, output_raster_path=bin_cf, output_data_type=gdal.GDT_Float32, output_ndv=-9999.0, verbose=True)
                        ds_b = gdal.Open(bin_cf, gdal.GA_ReadOnly)
                        b = ds_b.GetRasterBand(1)
                        stats = b.GetStatistics(False, True)
                        if stats is not None:
                            bmin, bmax = float(stats[0]), float(stats[1])
                            if bmin < -1e-6 or bmax > 1.0 + 1e-6:
                                raise ValueError(f'Binary raster values outside [0,1]: min={bmin}, max={bmax} ({bin_cf})')
                        s3_handler.upload_from_temp(os.path.join(out_base, os.path.basename(bin_cf)), bin_cf)
                        hb.warp_raster_to_match(input_path=str(bin_cf), output_path=share_cf, match_path=p.reference_raster_path, resample_method='average', src_ndv=-9999.0, dst_ndv=-9999.0, output_data_type=gdal.GDT_Float32)
                        ds_s = gdal.Open(share_cf, gdal.GA_ReadOnly)
                        bs = ds_s.GetRasterBand(1)
                        stats_s = bs.GetStatistics(False, True)
                        smin, smax = float(stats_s[0]), float(stats_s[1])
                        if smin < -1e-6 or smax > 1.0 + 1e-6:
                            raise ValueError(f'Share raster values outside [0,1]: min={smin}, max={smax} ({share_cf})')
                        else:
                            hb.log(f'[{mode_name}] Counterfactual share raster for {cat} {year} has valid range: min={smin}, max={smax}')
                        s3_handler.upload_from_temp(s3_share_cf, share_cf)
                        hb.log(f'Created [{mode_name}] counterfactual share raster for {cat} {year} at {s3_share_cf}')

    return p


def preprocess_deforestation_from_esacci(p):
    if not getattr(p, 'run_this', True):
        return p

    years = list(p.time_range)
    if not years:
        raise ValueError('p.time_range is empty; cannot build deforestation exposures.')

    source_bases = {
        'coarse': os.path.join(*p.base_data, 'preprocess_data', 'esa'),
        'refined': os.path.join(*p.base_data, 'preprocess_data', 'esa_refined'),
    }
    output_bases = {
        'coarse': os.path.join(*p.base_data, 'preprocess_data', 'deforestation', 'coarse'),
        'refined': os.path.join(*p.base_data, 'preprocess_data', 'deforestation', 'refined'),
    }
    specs = {
        'coarse': [('forest_share', 'esacci_share_forest_{year}.tif')],
        'refined': [
            ('forest_dense_share_refined', 'esacci_share_refined_forest_dense_{year}.tif'),
            ('forest_open_share_refined', 'esacci_share_refined_forest_open_{year}.tif'),
        ],
    }
    windows = [1, 3]
    nodata = -9999.0

    def _src_path(kind, source_key, year):
        return os.path.join(source_bases[kind], source_key.format(year=year))

    def _out_path(kind, source_name, window, year):
        return os.path.join(output_bases[kind], f'deforestation_{source_name}_{window}yr_{year}.tif')

    needed_sources = []
    for kind, entries in specs.items():
        for source_name, source_key in entries:
            for year in years:
                needed_sources.append(_src_path(kind, source_key, year))

    missing_sources = [path for path in needed_sources if not s3_handler.file_exists(path)]
    if missing_sources:
        p.L.info('ESA share rasters missing; generating them before deforestation preprocessing.')
        preprocess_esacci_to_share(p)
        missing_sources = [path for path in needed_sources if not s3_handler.file_exists(path)]
        if missing_sources:
            raise FileNotFoundError(
                'Required ESA share rasters are still missing after preprocessing. '
                f'Examples: {missing_sources[:5]}'
            )

    out_paths = []
    for kind, entries in specs.items():
        for source_name, _source_key in entries:
            for window in windows:
                for year in years:
                    out_paths.append(_out_path(kind, source_name, window, year))

    if all(s3_handler.file_exists(path) for path in out_paths) and not getattr(p, 'force_run', False):
        p.L.info('All deforestation rasters already exist (coarse + refined, 1yr + 3yr).')
        return p

    p.L.info('Generating deforestation rasters from ESA CCI share products...')
    p.L.info('Exposure windows: 1-year loss and 3-year net loss (current vs. lagged forest share).')

    def _write_empty_like(template_path, out_path):
        ds_in = gdal.Open(template_path, gdal.GA_ReadOnly)
        if ds_in is None:
            raise RuntimeError(f'Could not open template raster: {template_path}')
        driver = gdal.GetDriverByName('GTiff')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        ds_out = driver.Create(
            out_path,
            ds_in.RasterXSize,
            ds_in.RasterYSize,
            1,
            gdal.GDT_Float32,
            options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'],
        )
        ds_out.SetGeoTransform(ds_in.GetGeoTransform())
        ds_out.SetProjection(ds_in.GetProjection())
        band_out = ds_out.GetRasterBand(1)
        band_out.SetNoDataValue(nodata)
        band_out.Fill(nodata)
        band_out.FlushCache()
        ds_out.FlushCache()
        ds_out = None
        ds_in = None

    def _write_loss_raster(prev_path, curr_path, out_path, label):
        ds_prev = gdal.Open(prev_path, gdal.GA_ReadOnly)
        ds_curr = gdal.Open(curr_path, gdal.GA_ReadOnly)
        if ds_prev is None:
            raise RuntimeError(f'Could not open previous-year raster: {prev_path}')
        if ds_curr is None:
            raise RuntimeError(f'Could not open current-year raster: {curr_path}')
        if ds_prev.RasterXSize != ds_curr.RasterXSize or ds_prev.RasterYSize != ds_curr.RasterYSize:
            raise ValueError(f'Raster size mismatch for {label}: {prev_path} vs {curr_path}')

        prev_band = ds_prev.GetRasterBand(1)
        curr_band = ds_curr.GetRasterBand(1)
        prev_ndv = prev_band.GetNoDataValue()
        curr_ndv = curr_band.GetNoDataValue()
        driver = gdal.GetDriverByName('GTiff')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        ds_out = driver.Create(
            out_path,
            ds_curr.RasterXSize,
            ds_curr.RasterYSize,
            1,
            gdal.GDT_Float32,
            options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'],
        )
        ds_out.SetGeoTransform(ds_curr.GetGeoTransform())
        ds_out.SetProjection(ds_curr.GetProjection())
        out_band = ds_out.GetRasterBand(1)
        out_band.SetNoDataValue(nodata)

        block_xsize, block_ysize = curr_band.GetBlockSize()
        if block_xsize <= 0 or block_ysize <= 0:
            block_xsize, block_ysize = 512, 512

        total_valid = 0
        total_sum = 0.0
        total_max = 0.0
        rows = ds_curr.RasterYSize
        cols = ds_curr.RasterXSize
        for yoff in range(0, rows, block_ysize):
            ysize = min(block_ysize, rows - yoff)
            for xoff in range(0, cols, block_xsize):
                xsize = min(block_xsize, cols - xoff)
                prev_arr = prev_band.ReadAsArray(xoff, yoff, xsize, ysize).astype(np.float32)
                curr_arr = curr_band.ReadAsArray(xoff, yoff, xsize, ysize).astype(np.float32)
                valid = np.isfinite(prev_arr) & np.isfinite(curr_arr)
                if prev_ndv is not None:
                    valid &= prev_arr != prev_ndv
                if curr_ndv is not None:
                    valid &= curr_arr != curr_ndv

                loss = np.full(prev_arr.shape, nodata, dtype=np.float32)
                if np.any(valid):
                    diff = np.clip(prev_arr[valid] - curr_arr[valid], 0.0, 1.0)
                    loss[valid] = diff.astype(np.float32)
                    total_valid += int(valid.sum())
                    total_sum += float(np.sum(diff))
                    block_max = float(np.max(diff))
                    if block_max > total_max:
                        total_max = block_max

                out_band.WriteArray(loss, xoff=xoff, yoff=yoff)

        out_band.FlushCache()
        ds_out.FlushCache()
        ds_out = None
        ds_prev = None
        ds_curr = None
        mean_loss = (total_sum / total_valid) if total_valid else 0.0
        p.L.info(f'[{label}] valid_pixels={total_valid:,} mean_loss={mean_loss:.6f} max_loss={total_max:.6f}')

    with s3_handler.temp_workspace('deforestation') as temp_dir:
        source_cache = {}

        def _get_local_source(kind, source_key, year):
            s3_path = _src_path(kind, source_key, year)
            if s3_path not in source_cache:
                source_cache[s3_path] = str(s3_handler.download_to_temp(s3_path, filename=os.path.basename(s3_path)))
            return source_cache[s3_path]

        for kind, entries in specs.items():
            for source_name, source_key in entries:
                for year in years:
                    rel_name_1yr = os.path.join(kind, f'deforestation_{source_name}_1yr_{year}.tif')
                    rel_name_3yr = os.path.join(kind, f'deforestation_{source_name}_3yr_{year}.tif')
                    s3_out_1yr = _out_path(kind, source_name, 1, year)
                    s3_out_3yr = _out_path(kind, source_name, 3, year)
                    local_out_1yr = os.path.join(temp_dir, rel_name_1yr)
                    local_out_3yr = os.path.join(temp_dir, rel_name_3yr)
                    os.makedirs(os.path.dirname(local_out_1yr), exist_ok=True)
                    os.makedirs(os.path.dirname(local_out_3yr), exist_ok=True)

                    curr_local = _get_local_source(kind, source_key, year)

                    prev_year_1 = year - 1
                    if prev_year_1 < years[0]:
                        _write_empty_like(curr_local, local_out_1yr)
                        p.L.info(f'[{kind}] {source_name} {year}: no 1-year lag available; wrote nodata raster.')
                    else:
                        prev_local = _get_local_source(kind, source_key, prev_year_1)
                        _write_loss_raster(prev_local, curr_local, local_out_1yr, f'{kind}:{source_name}:1yr:{year}')

                    prev_year_3 = year - 3
                    if prev_year_3 < years[0]:
                        _write_empty_like(curr_local, local_out_3yr)
                        p.L.info(f'[{kind}] {source_name} {year}: no 3-year lag available; wrote nodata raster.')
                    else:
                        prev_local = _get_local_source(kind, source_key, prev_year_3)
                        _write_loss_raster(prev_local, curr_local, local_out_3yr, f'{kind}:{source_name}:3yr:{year}')

                    s3_handler.upload_from_temp(s3_out_1yr, rel_name_1yr)
                    s3_handler.upload_from_temp(s3_out_3yr, rel_name_3yr)
                    p.L.info(f'[{kind}] saved deforestation outputs for {source_name} {year} -> {s3_out_1yr}, {s3_out_3yr}')

    p.L.info('Deforestation preprocessing complete.')
    return p

# def preprocess_esacci_to_share(p):
#     """
#     Preprocess ESA CCI land cover data to create annual vegetation share rasters and counterfactual vegetation scenarios.
#     1. Load raw ESA CCI land cover data
#         format: GeoTIFF raster
#         source: 
#         citation: 
#     2. For each year, calculate vegetation share (proportion of vegetated land cover types)
#     3. Create counterfactual vegetation scenarios (e.g., no natural vegetation)
#     4. Reproject and align with reference
#     """
#     if p.run_this:
#         correspondence_path = os.path.join(p.base_data_dir, 'esa_seals7_correspondence.csv')
#         corr_df = pd.read_csv(correspondence_path)
#         # Clean potential duplicate header rows or non-numeric src_id values
#         corr_df = corr_df[corr_df['src_id'].apply(lambda x: str(x).strip().isdigit())].copy()
#         corr_df['src_id'] = corr_df['src_id'].astype(int)
#         # Build mapping: coarse category_name -> set of src_id
#         categories = corr_df['dst_label'].unique()
#         # Ensure src_id keys are integers to avoid mixed-type dict keys
#         category_code_map = {cat: set(int(x) for x in corr_df[corr_df['dst_label'] == cat]['src_id']) for cat in categories}
        
#         # Build mapping for cf: all noveg become bare_areas
#         noveg_labels = ['forest', 'cropland', 'othernat']
#         category_noveg_code_map = {}
#         for cat in categories:
#             if cat in noveg_labels:
#                 # All src_ids for this category are remapped to bare_areas
#                 category_noveg_code_map[cat] = set(int(x) for x in corr_df[corr_df['dst_label'] == 'bare_areas']['src_id'])
#             else:
#                 # Keep original src_ids for other categories
#                 category_noveg_code_map[cat] = set(int(x) for x in corr_df[corr_df['dst_label'] == cat]['src_id'])
        

#         # Build refined mapping if present: dst_label_refined -> set of src_id
#         refined_col = 'dst_label_refined'
#         refined_categories = []
#         refined_code_map = {}
#         if refined_col in corr_df.columns:
#             refined_categories = corr_df[refined_col].unique()
#             refined_code_map = {cat: set(int(x) for x in corr_df[corr_df[refined_col] == cat]['src_id']) for cat in refined_categories}

#         refined_noveg_labels = [
#             'forest_dense',
#             'forest_open',
#             'cropland_rainfed',
#             'cropland_irrigated',
#             'grassland',
#             'shrubland',
#             'sparse',
#         ]
#         refined_noveg_code_map = {}
#         for cat in refined_categories:
#             if cat in refined_noveg_labels:
#                 refined_noveg_code_map[cat] = set(
#                     int(x) for x in corr_df[corr_df['dst_label'] == 'bare_areas']['src_id']
#                 )
#             else:
#                 refined_noveg_code_map[cat] = set(
#                     int(x) for x in corr_df[corr_df[refined_col] == cat]['src_id']
#                 )

#         # Output paths for each coarse category and year (observed and cf)
#         obs_paths = {cat: [os.path.join(*p.base_data, 'preprocess_data', 'esa', f'esacci_share_{cat}_{year}.tif') for year in p.time_range] for cat in categories}
#         cf_paths = {cat: [os.path.join(*p.base_data, 'preprocess_data', 'esa', f'esacci_share_cf_{cat}_{year}.tif') for year in p.time_range] for cat in categories}

#         # Output paths for refined categories (if any)
#         if len(refined_categories) > 0:
#             obs_paths_refined = {cat: [os.path.join(*p.base_data, 'preprocess_data', 'esa_refined', f'esacci_share_refined_{cat}_{year}.tif') for year in p.time_range] for cat in refined_categories}
#             cf_paths_refined = {cat: [os.path.join(*p.base_data, 'preprocess_data', 'esa_refined', f'esacci_share_refined_cf_{cat}_{year}.tif') for year in p.time_range] for cat in refined_categories}
#         else:
#             obs_paths_refined = {}
#             cf_paths_refined = {}

#         # Check if all observed and cf outputs exist (coarse + refined)
#         all_obs_exist_coarse = all(s3_handler.file_exists(path) for cat in categories for path in obs_paths[cat])
#         all_cf_exist_coarse = all(s3_handler.file_exists(path) for cat in categories for path in cf_paths[cat])
#         all_obs_exist_refined = True if not obs_paths_refined else all(s3_handler.file_exists(path) for cat in obs_paths_refined for path in obs_paths_refined[cat])
#         all_cf_exist_refined = True if not cf_paths_refined else all(s3_handler.file_exists(path) for cat in cf_paths_refined for path in cf_paths_refined[cat])

#         if all_obs_exist_coarse and all_cf_exist_coarse and all_obs_exist_refined and all_cf_exist_refined:
#             p.L.info(f'All ESA CCI share rasters (observed and counterfactual) for coarse and refined categories already exist in {os.path.join(*p.base_data, "preprocess_data", "esa")}/')
#             return p
#         else:
#             p.L.info(f'Generating ESA CCI share rasters (observed and counterfactual) for coarse and refined categories in {os.path.join(*p.base_data, "preprocess_data", "esa")}/')

#         for year in p.time_range:
#             raw_path = os.path.join(p.user_dir, 'Files', 'base_data', 'lulc', 'esa', f'lulc_esa_{year}.tif')
#             with s3_handler.temp_workspace('esa_share') as temp_dir:
#                 for cat in categories:
#                     if cat == '6':
#                         p.L.info(f'Skipping category 6 (water) for {year}.')
#                         continue
#                     # Observed raster
#                     binary_path_obs = os.path.join(temp_dir, f'esacci_binary_{cat}_{year}.tif')
#                     share_path_obs = os.path.join(temp_dir, f'esacci_share_{cat}_{year}.tif')
#                     s3_binary_out_path_obs = os.path.join(*p.base_data, 'preprocess_data', 'esa', f'esacci_binary_{cat}_{year}.tif')
#                     s3_share_out_path_obs = os.path.join(*p.base_data, 'preprocess_data', 'esa', f'esacci_share_{cat}_{year}.tif')
#                     # Counterfactual raster
#                     binary_path_cf = os.path.join(temp_dir, f'esacci_binary_cf_{cat}_{year}.tif')
#                     share_path_cf = os.path.join(temp_dir, f'esacci_share_cf_{cat}_{year}.tif')
#                     s3_binary_out_path_cf = os.path.join(*p.base_data, 'preprocess_data', 'esa', f'esacci_binary_cf_{cat}_{year}.tif')
#                     s3_share_out_path_cf = os.path.join(*p.base_data, 'preprocess_data', 'esa', f'esacci_share_cf_{cat}_{year}.tif')
#                     # Only generate observed raster if missing
#                     all_src_ids = set(int(x) for x in corr_df['src_id'])
#                     value_map_obs = {int(src_id): (1 if int(src_id) in category_code_map[cat] else 0) for src_id in all_src_ids}
#                     value_map_obs[255] = 0
#                     if not s3_handler.file_exists(s3_share_out_path_obs):
#                         p.L.debug(f'Reclassification rules (observed) for {cat} {year}: {value_map_obs}')
#                         hb.reclassify_raster_hb(
#                             input_raster_path=raw_path,
#                             rules=value_map_obs,
#                             output_raster_path=binary_path_obs,
#                             output_data_type=gdal.GDT_Float32,
#                             output_ndv=-9999.0,
#                             verbose=True
#                         )
#                         # Sanity-check binary raster stats before upload
#                         try:
#                             ds_b = gdal.Open(binary_path_obs, gdal.GA_ReadOnly)
#                             b = ds_b.GetRasterBand(1)
#                             stats = b.GetStatistics(False, True)
#                         except Exception as e:
#                             raise RuntimeError(f'Error checking binary raster stats for {binary_path_obs}: {e}')
#                         bmin, bmax = float(stats[0]), float(stats[1])
#                         if bmin < -1e-6 or bmax > 1.0 + 1e-6:
#                             raise ValueError(f'Binary raster values outside [0,1]: min={bmin}, max={bmax} ({binary_path_obs})')
#                         s3_handler.upload_from_temp(s3_binary_out_path_obs, binary_path_obs)
#                         hb.warp_raster_to_match(
#                             input_path=str(binary_path_obs),
#                             output_path=share_path_obs,
#                             match_path=p.reference_raster_path,
#                             resample_method='average',
#                             src_ndv=-9999.0,
#                             dst_ndv=-9999.0,
#                             output_data_type=gdal.GDT_Float32,
#                         )
#                         # Sanity-check warped share raster before upload (should be within [0,1])
#                         try:
#                             ds_s = gdal.Open(share_path_obs, gdal.GA_ReadOnly)
#                             bs = ds_s.GetRasterBand(1)
#                             stats_s = bs.GetStatistics(False, True)
#                         except Exception as e:
#                             raise RuntimeError(f'Error checking share raster stats for {share_path_obs}: {e}')
#                         smin, smax = float(stats_s[0]), float(stats_s[1])
#                         if smin < -1e-6 or smax > 1.0 + 1e-6:
#                             raise ValueError(f'Share raster values outside [0,1]: min={smin}, max={smax} ({share_path_obs})')
#                         s3_handler.upload_from_temp(s3_share_out_path_obs, share_path_obs)
#                         p.L.info(f'Created ESA CCI observed share raster for {cat} {year} at {s3_share_out_path_obs}')
#                     # Only generate counterfactual raster if missing
#                     value_map_cf = {int(src_id): (1 if int(src_id) in category_noveg_code_map[cat] else 0) for src_id in all_src_ids}
#                     value_map_cf[255] = 0
#                     if not s3_handler.file_exists(s3_share_out_path_cf):
#                         p.L.info(f'Reclassification rules (counterfactual) for {cat} {year}: {value_map_cf}')
#                         hb.reclassify_raster_hb(
#                             input_raster_path=raw_path,
#                             rules=value_map_cf,
#                             output_raster_path=binary_path_cf,
#                             output_data_type=gdal.GDT_Float32,
#                             output_ndv=-9999.0,
#                             verbose=True
#                         )
#                         try:
#                             ds_b = gdal.Open(binary_path_cf, gdal.GA_ReadOnly)
#                             b = ds_b.GetRasterBand(1)
#                             stats = b.GetStatistics(False, True)
#                         except Exception as e:
#                             raise RuntimeError(f'Error checking binary raster stats for {binary_path_cf}: {e}')
#                         bmin, bmax = float(stats[0]), float(stats[1])
#                         if bmin < -1e-6 or bmax > 1.0 + 1e-6:
#                             raise ValueError(f'Binary raster values outside [0,1]: min={bmin}, max={bmax} ({binary_path_cf})')
#                         s3_handler.upload_from_temp(s3_binary_out_path_cf, binary_path_cf)
#                         hb.warp_raster_to_match(
#                             input_path=str(binary_path_cf),
#                             output_path=share_path_cf,
#                             match_path=p.reference_raster_path,
#                             resample_method='average',
#                             src_ndv=-9999.0,
#                             dst_ndv=-9999.0,
#                             output_data_type=gdal.GDT_Float32,
#                         )
#                         try:
#                             ds_s = gdal.Open(share_path_cf, gdal.GA_ReadOnly)
#                             bs = ds_s.GetRasterBand(1)
#                             stats_s = bs.GetStatistics(False, True)
#                         except Exception as e:
#                             raise RuntimeError(f'Error checking share raster stats for {share_path_cf}: {e}')
#                         smin, smax = float(stats_s[0]), float(stats_s[1])
#                         if smin < -1e-6 or smax > 1.0 + 1e-6:
#                             raise ValueError(f'Share raster values outside [0,1]: min={smin}, max={smax} ({share_path_cf})')
#                         s3_handler.upload_from_temp(s3_share_out_path_cf, share_path_cf)
#                         p.L.info(f'Created ESA CCI counterfactual share raster for {cat} {year} at {s3_share_out_path_cf}')
#                 # --- Now also generate refined share rasters if mapping present ---
#                 if len(refined_categories) > 0:
#                     for rcat in refined_categories:
#                         # Observed raster (refined)
#                         binary_path_obs_r = os.path.join(temp_dir, f'esacci_binary_refined_{rcat}_{year}.tif')
#                         share_path_obs_r = os.path.join(temp_dir, f'esacci_share_refined_{rcat}_{year}.tif')
#                         s3_binary_out_path_obs_r = os.path.join(*p.base_data, 'preprocess_data', 'esa_refined', f'esacci_binary_refined_{rcat}_{year}.tif')
#                         s3_share_out_path_obs_r = os.path.join(*p.base_data, 'preprocess_data', 'esa_refined', f'esacci_share_refined_{rcat}_{year}.tif')
#                         # Counterfactual raster (refined)
#                         binary_path_cf_r = os.path.join(temp_dir, f'esacci_binary_refined_cf_{rcat}_{year}.tif')
#                         share_path_cf_r = os.path.join(temp_dir, f'esacci_share_refined_cf_{rcat}_{year}.tif')
#                         s3_binary_out_path_cf_r = os.path.join(*p.base_data, 'preprocess_data', 'esa_refined', f'esacci_binary_refined_cf_{rcat}_{year}.tif')
#                         s3_share_out_path_cf_r = os.path.join(*p.base_data, 'preprocess_data', 'esa_refined', f'esacci_share_refined_cf_{rcat}_{year}.tif')

#                         # Build rules for observed refined category
#                         all_src_ids = set(int(x) for x in corr_df['src_id'])
#                         value_map_obs_r = {int(src_id): (1 if int(src_id) in refined_code_map[rcat] else 0) for src_id in all_src_ids}
#                         value_map_obs_r[255] = 0
#                         if not s3_handler.file_exists(s3_share_out_path_obs_r):
#                             p.L.debug(f'Reclassification rules (observed refined) for {rcat} {year}: {value_map_obs_r}')
#                             hb.reclassify_raster_hb(
#                                 input_raster_path=raw_path,
#                                 rules=value_map_obs_r,
#                                 output_raster_path=binary_path_obs_r,
#                                 output_data_type=gdal.GDT_Float32,
#                                 output_ndv=-9999.0,
#                                 verbose=True
#                             )
#                             try:
#                                 ds_b = gdal.Open(binary_path_obs_r, gdal.GA_ReadOnly)
#                                 b = ds_b.GetRasterBand(1)
#                                 stats = b.GetStatistics(False, True)
#                             except Exception as e:
#                                 raise RuntimeError(f'Error checking binary raster stats for {binary_path_obs_r}: {e}')
#                             bmin, bmax = float(stats[0]), float(stats[1])
#                             if bmin < -1e-6 or bmax > 1.0 + 1e-6:
#                                 raise ValueError(f'Binary refined raster values outside [0,1]: min={bmin}, max={bmax} ({binary_path_obs_r})')
#                             s3_handler.upload_from_temp(s3_binary_out_path_obs_r, binary_path_obs_r)
#                             hb.warp_raster_to_match(
#                                 input_path=str(binary_path_obs_r),
#                                 output_path=share_path_obs_r,
#                                 match_path=p.reference_raster_path,
#                                 resample_method='average',
#                                 src_ndv=-9999.0,
#                                 dst_ndv=-9999.0,
#                                 output_data_type=gdal.GDT_Float32,
#                             )
#                             try:
#                                 ds_s = gdal.Open(share_path_obs_r, gdal.GA_ReadOnly)
#                                 bs = ds_s.GetRasterBand(1)
#                                 stats_s = bs.GetStatistics(False, True)
#                             except Exception as e:
#                                 raise RuntimeError(f'Error checking share raster stats for {share_path_obs_r}: {e}')
#                             smin, smax = float(stats_s[0]), float(stats_s[1])
#                             if smin < -1e-6 or smax > 1.0 + 1e-6:
#                                 raise ValueError(f'Refined share raster values outside [0,1]: min={smin}, max={smax} ({share_path_obs_r})')
#                             s3_handler.upload_from_temp(s3_share_out_path_obs_r, share_path_obs_r)
#                             p.L.info(f'Created ESA CCI observed REFINED share raster for {rcat} {year} at {s3_share_out_path_obs_r}')

#                         # Build rules for counterfactual refined category (noveg -> bare_areas)
#                         value_map_cf_r = {int(src_id): (1 if int(src_id) in refined_noveg_code_map.get(rcat, set()) else 0) for src_id in all_src_ids}
#                         value_map_cf_r[255] = 0
#                         if not s3_handler.file_exists(s3_share_out_path_cf_r):
#                             p.L.debug(f'Reclassification rules (counterfactual refined) for {rcat} {year}: {value_map_cf_r}')
#                             hb.reclassify_raster_hb(
#                                 input_raster_path=raw_path,
#                                 rules=value_map_cf_r,
#                                 output_raster_path=binary_path_cf_r,
#                                 output_data_type=gdal.GDT_Float32,
#                                 output_ndv=-9999.0,
#                                 verbose=True
#                             )
#                             try:
#                                 ds_b = gdal.Open(binary_path_cf_r, gdal.GA_ReadOnly)
#                                 b = ds_b.GetRasterBand(1)
#                                 stats = b.GetStatistics(False, True)
#                                 if stats is not None:
#                                     bmin, bmax = float(stats[0]), float(stats[1])
#                                     if bmin < -1e-6 or bmax > 1.0 + 1e-6:
#                                         raise ValueError(f'Binary refined raster values outside [0,1]: min={bmin}, max={bmax} ({binary_path_cf_r})')
#                             finally:
#                                 try:
#                                     ds_b = None
#                                 except Exception:
#                                     pass
#                             s3_handler.upload_from_temp(s3_binary_out_path_cf_r, binary_path_cf_r)
#                             hb.warp_raster_to_match(
#                                 input_path=str(binary_path_cf_r),
#                                 output_path=share_path_cf_r,
#                                 match_path=p.reference_raster_path,
#                                 resample_method='average',
#                                 src_ndv=-9999.0,
#                                 dst_ndv=-9999.0,
#                                 output_data_type=gdal.GDT_Float32,
#                             )
#                             try:
#                                 ds_s = gdal.Open(share_path_cf_r, gdal.GA_ReadOnly)
#                                 bs = ds_s.GetRasterBand(1)
#                                 stats_s = bs.GetStatistics(False, True)
#                             except Exception as e:
#                                 raise RuntimeError(f'Error checking share raster stats for {share_path_cf_r}: { e}')
#                             smin, smax = float(stats_s[0]), float(stats_s[1])
#                             if smin < -1e-6 or smax > 1.0 + 1e-6:
#                                     raise ValueError(f'Refined share raster values outside [0,1]: min={smin}, max={smax} ({share_path_cf_r})')
#                             s3_handler.upload_from_temp(s3_share_out_path_cf_r, share_path_cf_r)
#                             p.L.info(f'Created ESA CCI counterfactual REFINED share raster for {rcat} {year} at {s3_share_out_path_cf_r}')

def preprocess_sdr(p):
    """
    Preprocess InVEST SDR avoided erosion rasters for observed and
    counterfactual (noveg) scenarios, warping to the project reference grid.

    The avoided erosion raster represents:
        avoided_erosion = sediment_export_noveg - sediment_export_observed

    Positive values mean vegetation is retaining sediment that would otherwise
    be mobilised — directly interpretable as an ecosystem service.

    Source structure (S3)
    ---------------------
    {base_data}/invest_sdr_output/lulc_esa_{year}/avoided_erosion.tif
    {base_data}/invest_sdr_output/lulc_esa_{year}_cf/avoided_erosion.tif

    Outputs (S3)
    ------------
    {base_data}/preprocess_data/sdr/avoided_erosion_observed_{year}.tif
    {base_data}/preprocess_data/sdr/avoided_erosion_counterfactual_{year}.tif

    Processing
    ----------
    For each year and scenario:
        1. Download raw avoided_erosion.tif from S3
        2. Warp / align to project reference raster (bilinear resampling)
        3. Clip negative values to 0 (erosion increase under observed vs noveg
           is not part of the ES definition)
        4. Cast to float32 and set nodata to NaN
        5. Upload to S3 output location
    """
    if not p.run_this:
        return p

    # ------------------------------------------------------------------
    # Check if all outputs already exist
    # ------------------------------------------------------------------
    out_paths = []
    for year in p.time_range:
        for scenario in ['observed', 'counterfactual']:
            out_paths.append(
                os.path.join(*p.base_data, 'preprocess_data', 'sdr',
                             f'avoided_erosion_{scenario}_{year}.tif')
            )

    if all(s3_handler.file_exists(p) for p in out_paths):
        hb.log('All SDR avoided erosion rasters already exist on S3, skipping.')
        return p

    # ------------------------------------------------------------------
    # Process each year
    # ------------------------------------------------------------------
    for year in p.time_range:

        scenarios = {
            'observed':        f'lulc_esa_{year}',
            'counterfactual':  f'lulc_esa_{year}_cf',
        }

        for scenario, folder in scenarios.items():
            s3_out = os.path.join(*p.base_data, 'preprocess_data', 'sdr',
                                  f'avoided_erosion_{scenario}_{year}.tif')

            if s3_handler.file_exists(s3_out):
                hb.log(f'  Already exists, skipping: avoided_erosion_{scenario}_{year}.tif')
                continue

            s3_raw = os.path.join(*p.base_data, 'invest_sdr_output',
                                  folder, 'avoided_erosion.tif')

            if not s3_handler.file_exists(s3_raw):
                hb.log(f'  WARNING: source not found, skipping: {s3_raw}')
                continue

            hb.log(f'  Processing SDR {scenario} {year}...')

            with s3_handler.temp_workspace(f'sdr_{scenario}_{year}') as temp_dir:

                # 1. Download
                local_raw    = s3_handler.download_to_temp(s3_raw)
                local_warped = os.path.join(temp_dir, 'warped.tif')
                local_out    = os.path.join(temp_dir,
                                            f'avoided_erosion_{scenario}_{year}.tif')

                # 2. Warp to reference grid
                hb.warp_raster_to_match(
                    input_path=str(local_raw),
                    output_path=local_warped,
                    match_path=p.reference_raster_path,
                    resample_method='bilinear',
                    src_ndv=3.4028235e+38,  # SDR input nodata value
                    dst_ndv=-9999.0,
                )


                # 3. Load, clean, cast
                ds_in  = gdal.Open(local_warped, gdal.GA_ReadOnly)
                band   = ds_in.GetRasterBand(1)
                ndv    = band.GetNoDataValue()
                arr    = band.ReadAsArray().astype(np.float32)

                # Robustly mask all possible nodata sentinels as np.nan
                # - Mask -9999.0, 3.4028235e+38, band NDV, and any extremely large values (from resampling)
                arr[arr == np.float32(-9999.0)] = np.nan
                arr[arr == np.float32(3.4028235e+38)] = np.nan
                if ndv is not None and not np.isnan(ndv):
                    arr[arr == np.float32(ndv)] = np.nan
                # Mask any value > 1e20 (covers float32 max, resampling artifacts, etc.)
                arr[arr > 1e20] = np.nan
                arr[~np.isfinite(arr)] = np.nan

                # Clip negative values — avoided erosion < 0 means vegetation
                # increases erosion in that pixel, which we treat as 0 ES benefit
                n_negative = int(np.sum(arr < 0))
                if n_negative > 0:
                    hb.log(f'    Clipping {n_negative:,} negative avoided_erosion '
                           f'pixels to 0')
                    arr = np.where(arr < 0, 0.0, arr)

                # Log range for sanity check
                valid = arr[np.isfinite(arr)]
                if len(valid) > 0:
                    hb.log(f'    {scenario} {year}: '
                           f'min={float(valid.min()):.3f}  '
                           f'max={float(valid.max()):.3f}  '
                           f'mean={float(valid.mean()):.3f}  '
                           f'n_valid={len(valid):,}')

                # Convert np.nan to -9999.0 for output (file NDV)
                arr_out = np.where(np.isnan(arr), -9999.0, arr)

                # Post-write assertion: ensure only -9999.0 is present as nodata, no 3.4e+38 or NaN
                assert not np.any(arr_out == np.float32(3.4028235e+38)), "ERROR: 3.4e+38 still present in output!"
                assert not np.any(np.isnan(arr_out)), "ERROR: NaN present in output!"

                # 4. Write float32 GeoTIFF with NDV = -9999.0
                driver   = gdal.GetDriverByName('GTiff')
                n_cols_r = ds_in.RasterXSize
                n_rows_r = ds_in.RasterYSize
                ds_out   = driver.Create(
                    local_out, n_cols_r, n_rows_r, 1, gdal.GDT_Float32,
                    options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'],
                )
                ds_out.SetGeoTransform(ds_in.GetGeoTransform())
                ds_out.SetProjection(ds_in.GetProjection())
                band_out = ds_out.GetRasterBand(1)
                band_out.SetNoDataValue(-9999.0)
                band_out.WriteArray(arr_out)
                band_out.FlushCache()
                ds_out = ds_in = None

                # 5. Upload
                s3_handler.upload_from_temp(s3_out, local_out)
                hb.log(f'    Saved: {s3_out}')

    hb.log('SDR preprocessing complete.')
    return p


# def preprocess_grip_roads(p):
#     """
#     Compute road density (km of road per km²) at the project reference
#     resolution from GRIP4 global roads vector dataset.

#     All road types are combined into a single density surface — at 1 km
#     resolution the distinction between road classes matters less than total
#     linear infrastructure, which drives cut-slope and drainage-disruption
#     effects on landslide hazard.

#     Method
#     ------
#     1. Download GRIP4 .gdb from S3
#     2. Reproject to equal-area CRS (EPSG:6933) via ogr2ogr (streaming,
#        no Python memory pressure)
#     3. Segmentize to max pixel_m and compute length_m via ogr2ogr + SQLite
#        dialect (also streaming). Segmentizing ensures each sub-segment fits
#        within a single pixel so MERGE_ALG=ADD accumulates correct totals.
#     4. For each reference-grid pixel, sum road length (m) using rasterisation
#        with MERGE_ALG=ADD
#     5. Divide by pixel area (km²) to get road density (km/km²)
#     6. Warp result to reference grid (EPSG:4326)
#     7. Upload to S3

#     Key fix vs earlier version
#     --------------------------
#     Previously, length_m was computed on the full (un-segmentized) geometry
#     and then burned into every pixel the feature touched, causing massive
#     over-counting (e.g. a 6 km road burning 6000 m into all 6 pixels it
#     crossed rather than ~1000 m each). Segmentizing to pixel_m first ensures
#     each sub-segment only occupies one pixel, so MERGE_ALG=ADD accumulates
#     correct totals.

#     Source data
#     -----------
#     Global Roads Inventory Project (GRIP4):
#     "Global patterns of current and future road infrastructure"
#     Citation: Meijer et al. (2018). Nature, 555, 71-76.
#               https://doi.org/10.1038/nature25143
#     Download: https://zenodo.org/records/6420961

#     Source (S3)
#     -----------
#     {base_data}/GRIP4_global_vector_fgdb/GRIP4_GlobalRoads.gdb

#     Output (S3)
#     -----------
#     {base_data}/preprocess_data/grip_roads/road_density_km_per_km2.tif
#     """
#     if not p.run_this:
#         return p

#     s3_out = os.path.join(*p.base_data, 'preprocess_data', 'grip_roads',
#                           'road_density_km_per_km2.tif')

#     if s3_handler.file_exists(s3_out):
#         hb.log('Road density raster already exists on S3, skipping.')
#         return p

#     s3_source = os.path.join(*p.base_data, 'grip_roads', 'GRIP4_global_vector_fgdb',
#                              'GRIP4_GlobalRoads.gdb')
#     if not s3_handler.file_exists(s3_source):
#         hb.log(f'ERROR: GRIP4 source not found at {s3_source}')
#         return p

#     hb.log(f'Processing road density from {s3_source}...')

#     ogr2ogr_path = '/Users/mbraaksma/mambaforge/envs/teems_hb_dev/bin/ogr2ogr'
#     pixel_m = 1000.0  # 1 km pixels — matches reference raster

#     with s3_handler.temp_workspace('grip_roads') as temp_dir:

#         # ------------------------------------------------------------------
#         # 1. Download .gdb
#         # ------------------------------------------------------------------
#         local_src = s3_handler.download_to_temp(s3_source)
#         input_gdb = str(local_src)

#         # ------------------------------------------------------------------
#         # 2. Reproject to EPSG:6933 via ogr2ogr (fully streaming)
#         # ------------------------------------------------------------------
#         reprojected_gpkg = os.path.join(temp_dir, 'roads_ea.gpkg')
#         hb.log('  Pass 1: reprojecting to EPSG:6933 via ogr2ogr...')
#         subprocess.run([
#             ogr2ogr_path,
#             '-f', 'GPKG', reprojected_gpkg,
#             input_gdb,
#             '-t_srs', 'EPSG:6933',
#             '-progress',
#         ], check=True)
#         hb.log('  Pass 1 complete.')
#         subprocess.run([
#             ogr2ogr_path,
#             '-f', 'GPKG', reprojected_gpkg,
#             input_gdb,
#             'GRIP4_GlobalRoads',
#             '-t_srs', 'EPSG:6933',
#             '-progress',
#         ], check=True)
#         ds_debug = ogr.Open(reprojected_gpkg)
#         print('Number of layers:', ds_debug.GetLayerCount())
#         for i in range(ds_debug.GetLayerCount()):
#             lyr = ds_debug.GetLayer(i)
#             print(f'  Layer {i}: {lyr.GetName()}, {lyr.GetFeatureCount()} features')
#         ds_debug = None

#         # ------------------------------------------------------------------
#         # 3. Segmentize + compute length_m via ogr2ogr SQLite dialect
#         #
#         #    ST_Length is computed on the already-reprojected equal-area
#         #    geometry, so lengths are in metres. Segmentizing ensures no
#         #    sub-segment spans more than one pixel.
#         #
#         #    Layer and geometry column names are read dynamically to avoid
#         #    hardcoding assumptions about the GPKG internals.
#         # ------------------------------------------------------------------
#         ds_check   = ogr.Open(reprojected_gpkg)
#         lyr_check  = ds_check.GetLayer(0)
#         layer_name = lyr_check.GetName()
#         geom_col   = lyr_check.GetGeometryColumn() or 'geom'
#         ds_check   = None
#         hb.log(f'  Reprojected layer: "{layer_name}", geometry column: "{geom_col}"')

#         segmentized_gpkg = os.path.join(temp_dir, 'roads_ea_seg.gpkg')
#         hb.log('  Pass 2: segmentizing and computing length_m via ogr2ogr...')
#         subprocess.run([
#             ogr2ogr_path,
#             '-f', 'GPKG', segmentized_gpkg,
#             reprojected_gpkg,
#             '-segmentize', str(pixel_m),
#             '-dialect', 'SQLite',
#             '-sql', f'SELECT {geom_col}, ST_Length({geom_col}) AS length_m FROM "{layer_name}"',
#             '-progress',
#         ], check=True)
#         hb.log('  Pass 2 complete.')
#         # ds_debug = ogr.Open(segmentized_gpkg)
#         # lyr_debug = ds_debug.GetLayer(0)
#         # feat_debug = lyr_debug.GetNextFeature()
#         # print('First feature length_m:', feat_debug.GetField('length_m'))
#         # print('Geometry length:', feat_debug.GetGeometryRef().Length())
#         # print('All field names:', [lyr_debug.GetLayerDefn().GetFieldDefn(i).GetName() 
#         #                            for i in range(lyr_debug.GetLayerDefn().GetFieldCount())])
#         # ds_debug = None
#         ds_debug = ogr.Open(segmentized_gpkg)
#         lyr_debug = ds_debug.GetLayer(0)
#         lengths = []
#         for feat in lyr_debug:
#             lengths.append(feat.GetField('length_m'))
#         lengths.sort(reverse=True)
#         print('Top 10 lengths:', lengths[:10])
#         print('Total features:', len(lengths))
#         ds_debug = None

#         # ------------------------------------------------------------------
#         # 4. Rasterise road length (m) into an equal-area grid at pixel_m
#         # ------------------------------------------------------------------
#         # ds_vec = ogr.Open(segmentized_gpkg)
#         # layer  = ds_vec.GetLayer(0)

#         # minx, maxx, miny, maxy = layer.GetExtent()
#         # out_cols = int(np.ceil((maxx - minx) / pixel_m))
#         # out_rows = int(np.ceil((maxy - miny) / pixel_m))
#         # hb.log(f'  Equal-area grid: {out_cols} x {out_rows} at {pixel_m:.0f} m')
        
#         # Get extent from pass 1 reprojected GPKG — the SQLite dialect in
#         # pass 2 can strip spatial metadata, causing GetExtent() to return
#         # zeros and the rasterizer to burn nothing.
#         ds_ext = ogr.Open(reprojected_gpkg)
#         lyr_ext = ds_ext.GetLayer(0)
#         minx, maxx, miny, maxy = lyr_ext.GetExtent()
#         ds_ext = None
#         hb.log(f'  Layer extent: minx={minx:.0f}, maxx={maxx:.0f}, miny={miny:.0f}, maxy={maxy:.0f}')

#         ds_vec = ogr.Open(segmentized_gpkg)
#         layer  = ds_vec.GetLayer(0)

#         out_cols = int(np.ceil((maxx - minx) / pixel_m))
#         out_rows = int(np.ceil((maxy - miny) / pixel_m))
#         hb.log(f'  Equal-area grid: {out_cols} x {out_rows} at {pixel_m:.0f} m')


#         local_length_tif = os.path.join(temp_dir, 'road_length_m.tif')
#         tif_driver = gdal.GetDriverByName('GTiff')
#         ds_len = tif_driver.Create(
#             local_length_tif, out_cols, out_rows, 1, gdal.GDT_Float32,
#             options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'],
#         )
#         gt_ea = (minx, pixel_m, 0, maxy, 0, -pixel_m)
#         ds_len.SetGeoTransform(gt_ea)
#         srs_ea = osr.SpatialReference()
#         srs_ea.ImportFromEPSG(6933)
#         ds_len.SetProjection(srs_ea.ExportToWkt())
#         band_len = ds_len.GetRasterBand(1)
#         band_len.Fill(0.0)
#         band_len.SetNoDataValue(-9999.0)

#         hb.log('  Rasterising road lengths with MERGE_ALG=ADD...')
#         gdal.RasterizeLayer(
#             ds_len, [1], layer,
#             options=['ATTRIBUTE=length_m', 'MERGE_ALG=ADD'],
#         )
#         band_len.FlushCache()
#         ds_vec = layer = None

#         # Debug: upload intermediate length raster to inspect raw values
#         s3_length_out = os.path.join(*p.base_data, 'preprocess_data', 'grip_roads',
#                                       'road_length_m_debug.tif')
#         s3_handler.upload_from_temp(s3_length_out, local_length_tif)
#         hb.log(f'  Debug: uploaded raw length raster to {s3_length_out}')

#         # ------------------------------------------------------------------
#         # 5. Convert length (m) to density (km / km²)
#         #    pixel area = (pixel_m / 1000)² km²  →  1.0 km² at 1 km pixels
#         # ------------------------------------------------------------------
#         hb.log('  Computing density (km/km²)...')
#         len_arr   = band_len.ReadAsArray().astype(np.float32)
#         pixel_km2 = (pixel_m / 1000.0) ** 2
#         density   = np.where(len_arr > 0, (len_arr / 1000.0) / pixel_km2, 0.0)

#         local_density_ea = os.path.join(temp_dir, 'road_density_ea.tif')
#         ds_dens = tif_driver.Create(
#             local_density_ea, out_cols, out_rows, 1, gdal.GDT_Float32,
#             options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'],
#         )
#         ds_dens.SetGeoTransform(gt_ea)
#         ds_dens.SetProjection(srs_ea.ExportToWkt())
#         band_d = ds_dens.GetRasterBand(1)
#         band_d.SetNoDataValue(-9999.0)
#         band_d.WriteArray(density)
#         band_d.FlushCache()
#         ds_dens = ds_len = None

#         # ------------------------------------------------------------------
#         # 6. Warp equal-area density raster back to reference grid (EPSG:4326)
#         # ------------------------------------------------------------------
#         local_out = os.path.join(temp_dir, 'road_density_km_per_km2.tif')
#         hb.log('  Warping to reference grid...')
#         hb.warp_raster_to_match(
#             input_path=local_density_ea,
#             output_path=local_out,
#             match_path=p.reference_raster_path,
#             resample_method='bilinear',
#         )

#         # Bilinear resampling can introduce small negative values at edges;
#         # clamp everything to >= 0 and clear any stray nodata values.
#         ds_fix  = gdal.Open(local_out, gdal.GA_Update)
#         band_fx = ds_fix.GetRasterBand(1)
#         arr_fx  = band_fx.ReadAsArray().astype(np.float32)
#         arr_fx  = np.where((arr_fx < 0) | ~np.isfinite(arr_fx) | (arr_fx == -9999.0), 0.0, arr_fx)
#         band_fx.WriteArray(arr_fx)
#         band_fx.SetNoDataValue(-9999.0)
#         band_fx.FlushCache()
#         ds_fix = None

#         # ------------------------------------------------------------------
#         # 7. Upload
#         # ------------------------------------------------------------------
#         s3_handler.upload_from_temp(s3_out, local_out)
#         hb.log(f'Saved: {s3_out}')
#         road_px = arr_fx[arr_fx > 0]
#         if road_px.size:
#             hb.log(f'  Road density range: '
#                    f'{road_px.min():.4f} – {arr_fx.max():.4f} km/km²')

#     hb.log('Road density preprocessing complete.')
#     return p

def preprocess_grip_roads(p):
    """
    Warp pre-existing road density raster to the project reference grid.

    Uses nearest-neighbor resampling to avoid interpolation artifacts on
    density data. Includes validation to catch unreasonably high values
    that could cause numerical blow-ups in downstream modeling.
    """
    if not p.run_this:
        return p

    s3_out = os.path.join(*p.base_data, 'preprocess_data', 'grip_roads',
                          'road_density_km_per_km2.tif')

    if s3_handler.file_exists(s3_out) and not getattr(p, 'force_run', False):
        hb.log('Road density raster already exists on S3, skipping.')
        return p

    # Use the pre-existing large raster for testing
    s3_source = os.path.join(*p.base_data, 'grip_roads',
                             'GRIP4_density_total', 'grip4_total_dens_m_km2.asc')
    if not s3_handler.file_exists(s3_source):
        hb.log(f'ERROR: Source raster not found at {s3_source}')
        return p

    hb.log(f'Processing road density from {s3_source}...')

    with s3_handler.temp_workspace('grip_roads') as temp_dir:
        local_src = s3_handler.download_to_temp(s3_source)
        local_out = os.path.join(temp_dir, 'road_density_km_per_km2.tif')
        hb.log('  Warping to reference grid (nearest-neighbor to preserve density)...')
        hb.warp_raster_to_match(
            input_path=str(local_src),
            output_path=local_out,
            match_path=p.reference_raster_path,
            resample_method='nearest',
            src_ndv=-9999,
            dst_ndv=-9999,
            output_data_type=gdal.GDT_Float32,
        )

        # Replace all -9999 values with zero after warping and compute diagnostics
        ds_fix = gdal.Open(local_out, gdal.GA_Update)
        band_fx = ds_fix.GetRasterBand(1)
        arr_fx = band_fx.ReadAsArray().astype(np.float32)
        arr_fx = np.where((arr_fx < 0) | ~np.isfinite(arr_fx), 0.0, arr_fx)

        # Compute robust percentiles and clip to a high quantile (data-driven)
        p50, p95, p99, p999 = np.nanpercentile(arr_fx, [50, 95, 99, 99.9])
        cap_value = float(p99)  # clip to the 99th percentile as requested
        hb.log(f'  Road density percentiles (m/km²): 50={p50:.1f}, 95={p95:.1f}, 99={p99:.1f}, 99.9={p999:.1f}')

        # Clip extremes (meters per km²). This prevents numerical blow-ups downstream.
        arr_fx = np.clip(arr_fx, 0.0, cap_value)

        # Post-clip sanity check
        max_density = float(np.nanmax(arr_fx))
        assert max_density <= cap_value + 1e-6, (
            f'Post-clip road density max={max_density:.2f} m/km² exceeds cap={cap_value:.2f}.'
        )

        band_fx.WriteArray(arr_fx)
        band_fx.SetNoDataValue(-9999.0)  # declared but never present in data
        band_fx.ComputeStatistics(False)
        band_fx.FlushCache()
        ds_fix = None

        road_px = arr_fx[arr_fx > 0]
        if road_px.size:
            hb.log(
                f'  Road density range after clip (m/km²): {float(np.nanmin(road_px)):.4f} – {max_density:.4f}'
            )

        s3_handler.upload_from_temp(s3_out, local_out)
        hb.log(f'Saved: {s3_out}')

    hb.log('Road density warp test complete.')
    hb.log('Road density preprocessing complete.')
    return p


def preprocess_gem_faults(p):
    """
    Compute distance-to-nearest-active-fault (km) at the project reference
    resolution from the GEM Global Active Faults database.

    Only faults with high activity confidence (activity_confidence == 1)
    are used — these are faults with confirmed neotectonic activity.
    Fault proximity is a time-invariant landslide risk amplifier: fault
    zones have fractured, weakened rock that fails more readily under
    rainfall loading.

    Method
    ------
    1. Download GEM GeoJSON or GPKG from S3 (tries GPKG first)
    2. Filter to activity_confidence == 1
    3. Rasterise fault proximity: for each reference pixel, compute
       distance (km) to the nearest active fault polyline
    4. Warp result to reference grid
    5. Upload to S3

    Source data
    -----------
    GEM Global Active Faults Database (GEM-GAF):
    "GEMScienceTools/gem-global-active-faults: First release of 2019 (2019.0)"
    Author: Richard Styron
    Description: The GEM Global Active Faults Database is a global compilation of active fault data, harmonized and attributed for use in seismic hazard and risk modeling. The database includes fault geometry, slip rates, and activity confidence, and is provided in GeoJSON and GPKG formats. See Styron (2019) for details.
    Citation: Richard Styron. (2019). GEMScienceTools/gem-global-active-faults: First release of 2019 (2019.0). Zenodo. https://doi.org/10.5281/zenodo.3376300
    Download: https://zenodo.org/records/3376300

    Source (S3)
    -----------
    {base_data}/gem_faults/GEMScienceTools-gem-global-active-faults-03ad3ff/
        geojson/gem_active_faults_harmonized.gpkg   (preferred)
        geojson/gem_active_faults_harmonized.geojson (fallback)

    Output (S3)
    -----------
    {base_data}/preprocess_data/gem_faults/distance_to_fault_km.tif

    Notes on attribute filtering
    ----------------------------
    activity_confidence values:
        1 = confirmed neotectonic activity     ← used
        2 = likely active
        3 = possibly active
    Only confidence == 1 is used to avoid including uncertain faults that
    could add noise. The slip_type and slip_rate attributes are not used
    as the sign of their effect on landslide hazard is ambiguous at 1 km
    resolution and the data are sparse.
    """
    if not p.run_this:
        return p

    s3_out = os.path.join(*p.base_data, 'preprocess_data', 'gem_faults',
                          'distance_to_fault_km.tif')

    if s3_handler.file_exists(s3_out):
        hb.log('Fault distance raster already exists on S3, skipping.')
        return p

    # Try GPKG first, fall back to GeoJSON
    base_fault = os.path.join(
        *p.base_data,
        'gem_faults',
        'GEMScienceTools-gem-global-active-faults-03ad3ff',
        'geojson',
    )
    s3_gpkg    = os.path.join(base_fault, 'gem_active_faults_harmonized.gpkg')
    s3_geojson = os.path.join(base_fault, 'gem_active_faults_harmonized.geojson')

    if s3_handler.file_exists(s3_gpkg):
        s3_source = s3_gpkg
    elif s3_handler.file_exists(s3_geojson):
        s3_source = s3_geojson
    else:
        hb.log(f'ERROR: GEM faults source not found. Tried:\n  {s3_gpkg}\n  {s3_geojson}')
        return p

    hb.log(f'Processing fault distance from {s3_source}...')

    with s3_handler.temp_workspace('gem_faults') as temp_dir:

        # ------------------------------------------------------------------
        # 1. Download source
        # ------------------------------------------------------------------
        local_src = s3_handler.download_to_temp(s3_source)

        # ------------------------------------------------------------------
        # 2. Load and filter to confirmed active faults
        # ------------------------------------------------------------------
        import geopandas as gpd

        hb.log('  Loading GEM faults...')
        gdf = gpd.read_file(str(local_src))
        hb.log(f'  {len(gdf):,} total fault features')
        hb.log(f'  activity_confidence value counts:\n'
               f'{gdf["activity_confidence"].value_counts(dropna=False).to_string()}')

        # Filter to confirmed active (confidence == 1)
        if 'activity_confidence' in gdf.columns:
            gdf_active = gdf[gdf['activity_confidence'] == '1'].copy()
            hb.log(f'  {len(gdf_active):,} faults with activity_confidence == 1')
        else:
            hb.log('  WARNING: activity_confidence column not found — using all faults')
            gdf_active = gdf.copy()

        if len(gdf_active) == 0:
            hb.log('ERROR: No active faults found after filtering.')
            return p

        # ------------------------------------------------------------------
        # 3. Compute proximity raster
        #    Strategy: rasterise fault lines as 0, then compute Euclidean
        #    distance transform. This is much faster than per-pixel nearest-
        #    neighbour distance for a global 1 km grid.
        # ------------------------------------------------------------------

        # Reproject to equal-area for distance in metres
        EQUAL_AREA_CRS = 'EPSG:6933'
        hb.log(f'  Reprojecting to {EQUAL_AREA_CRS}...')
        gdf_ea = gdf_active.to_crs(EQUAL_AREA_CRS)

        # Save to temp GPKG for GDAL rasterisation
        local_gpkg_ea = os.path.join(temp_dir, 'faults_ea.gpkg')
        gdf_ea.to_file(local_gpkg_ea, driver='GPKG')

        # Build equal-area raster matching reference extent
        bounds_ea  = gdf_ea.total_bounds    # minx, miny, maxx, maxy
        # Expand bounds slightly to ensure full coverage
        pad_m      = 50000.0               # 50 km padding
        minx_ea    = bounds_ea[0] - pad_m
        miny_ea    = bounds_ea[1] - pad_m
        maxx_ea    = bounds_ea[2] + pad_m
        maxy_ea    = bounds_ea[3] + pad_m

        pixel_m    = 1000.0
        out_cols   = int(np.ceil((maxx_ea - minx_ea) / pixel_m))
        out_rows   = int(np.ceil((maxy_ea - miny_ea) / pixel_m))
        hb.log(f'  Equal-area proximity grid: {out_cols} x {out_rows}')

        # Create binary fault raster (1 = fault present, 0 = no fault)
        local_fault_bin = os.path.join(temp_dir, 'fault_binary_ea.tif')
        driver    = gdal.GetDriverByName('GTiff')
        srs_ea    = osr.SpatialReference()
        srs_ea.ImportFromEPSG(6933)

        ds_bin = driver.Create(
            local_fault_bin, out_cols, out_rows, 1, gdal.GDT_Byte,
            options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'],
        )
        gt_ea = (minx_ea, pixel_m, 0, maxy_ea, 0, -pixel_m)
        ds_bin.SetGeoTransform(gt_ea)
        ds_bin.SetProjection(srs_ea.ExportToWkt())
        band_bin = ds_bin.GetRasterBand(1)
        band_bin.Fill(0)
        band_bin.SetNoDataValue(255)

        ds_vec  = ogr.Open(local_gpkg_ea)
        layer   = ds_vec.GetLayer(0)
        gdal.RasterizeLayer(ds_bin, [1], layer, burn_values=[1])
        band_bin.FlushCache()
        ds_vec = layer = None

        # ------------------------------------------------------------------
        # 4. Euclidean distance transform on binary raster
        #    gdal.ComputeProximity gives distance to nearest non-zero pixel
        #    in the same units as the raster (metres here).
        # ------------------------------------------------------------------
        local_dist_ea = os.path.join(temp_dir, 'fault_distance_m_ea.tif')
        ds_dist = driver.Create(
            local_dist_ea, out_cols, out_rows, 1, gdal.GDT_Float32,
            options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'],
        )
        ds_dist.SetGeoTransform(gt_ea)
        ds_dist.SetProjection(srs_ea.ExportToWkt())
        band_dist = ds_dist.GetRasterBand(1)
        band_dist.SetNoDataValue(-9999.0)

        hb.log('  Computing proximity raster (this may take a few minutes)...')
        gdal.ComputeProximity(
            band_bin, band_dist,
            options=['VALUES=1', 'DISTUNITS=GEO'],   # GEO = same units as GT (metres)
        )
        band_dist.FlushCache()
        ds_bin = ds_dist = None

        # Convert metres to km
        ds_m   = gdal.Open(local_dist_ea, gdal.GA_Update)
        band_m = ds_m.GetRasterBand(1)
        arr_m  = band_m.ReadAsArray().astype(np.float32)
        arr_km = arr_m / 1000.0
        arr_km[arr_m == -9999.0] = np.nan
        band_m.WriteArray(arr_km)
        band_m.FlushCache()
        ds_m   = None

        # ------------------------------------------------------------------
        # 5. Warp to reference grid (EPSG:4326)
        # ------------------------------------------------------------------
        local_out = os.path.join(temp_dir, 'distance_to_fault_km.tif')
        hb.log('  Warping to reference grid...')
        hb.warp_raster_to_match(
            input_path=local_dist_ea,
            output_path=local_out,
            match_path=p.reference_raster_path,
            resample_method='bilinear',
        )


        # Set nodata (e.g. Antarctica) to zero in output raster
        ds_chk  = gdal.Open(local_out, gdal.GA_Update)
        band_chk = ds_chk.GetRasterBand(1)
        arr_chk = band_chk.ReadAsArray().astype(np.float32)
        ndv = band_chk.GetNoDataValue()
        if ndv is not None:
            arr_chk[arr_chk == ndv] = 0.0
            band_chk.WriteArray(arr_chk)
            band_chk.SetNoDataValue(0.0)
            band_chk.FlushCache()
        valid   = arr_chk[np.isfinite(arr_chk)]
        hb.log(f'  Distance to fault: min={float(valid.min()):.1f} km  '
            f'max={float(valid.max()):.1f} km  '
            f'mean={float(valid.mean()):.1f} km')
        ds_chk  = None

        # ------------------------------------------------------------------
        # 6. Upload
        # ------------------------------------------------------------------
        s3_handler.upload_from_temp(s3_out, local_out)
        hb.log(f'Saved: {s3_out}')

    hb.log('GEM fault distance preprocessing complete.')
    return p

# def convert_to_cog(p):
#     """
#     Convert all preprocessed rasters to Cloud Optimized GeoTIFFs (COGs) for efficient access in modeling and visualization.
#     1. For each preprocessed raster, check if COG version exists
#     2. If not, convert to COG using gdal with appropriate options for tiling, compression, and overviews
#     3. Upload COGs to S3
#     """
#     if not p.run_this:
#         return p

#     # List all preprocessed raster S3 prefixes (relative to bucket, not local paths)
#     # Assume p.base_data is a list of path components relative to the bucket root
#     preprocess_dirs = [
#         os.path.join(*p.base_data, 'preprocess_data', subdir)
#         for subdir in ['nasa_glc', 'ls_factor_gedtm', 'fao_gaez', 'esa', 'era5_land', 'worldpop']
#     ]

#     # Find all .tif rasters in these folders (S3 or local)
#     all_rasters = []
#     for d in preprocess_dirs:
#         p.L.info(f'Checking S3 prefix: {d}')
#         try:
#             files = [f for f in s3_handler.list_files(d, suffix='.tif')]
#             p.L.info(f'  Found {len(files)} .tif files in {d}')
#             all_rasters.extend(files)
#         except Exception as e:
#             p.L.error(f'  Error listing files in {d}: {e}')

#     p.L.info(f'Found {len(all_rasters)} preprocessed rasters to check for COG conversion (S3 only).')

#     for raster_path in all_rasters:
#         # COG output path: add _cog before .tif
#         if raster_path.endswith('.tif'):
#             cog_path = raster_path.replace('.tif', '_cog.tif')
#         else:
#             continue

#         # If COG already exists, skip
#         if s3_handler.file_exists(cog_path):
#             p.L.info(f'COG already exists for {raster_path}, skipping.')
#             continue

#         vsis3_in = s3_handler.get_vsis3_path(raster_path)
#         vsis3_out = s3_handler.get_vsis3_path(cog_path)
#         gdal.Translate(
#             vsis3_out, vsis3_in,
#             format='COG',
#             creationOptions=['COMPRESS=LZW', 'BLOCKSIZE=512', 'NUM_THREADS=ALL_CPUS']
#         )
#         p.L.info(f'COG written to {vsis3_out}')

#     p.L.info('COG conversion complete for all preprocessed rasters.')
#     return p